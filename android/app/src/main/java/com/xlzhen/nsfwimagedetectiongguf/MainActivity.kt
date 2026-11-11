package com.xlzhen.nsfwimagedetectiongguf

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.DocumentsContract
import android.provider.OpenableColumns
import android.util.LruCache
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.os.LocaleListCompat
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.color.MaterialColors
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.tabs.TabLayout
import com.xlzhen.nsfwimagedetectiongguf.databinding.ActivityMainBinding
import com.xlzhen.nsfwimagedetectiongguf.databinding.ItemResultBinding
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private val executor = Executors.newSingleThreadExecutor()
    private val thumbExecutor = Executors.newFixedThreadPool(2)
    private val main = Handler(Looper.getMainLooper())

    private data class Row(
        var uri: Uri,
        val name: String,
        val score: Float,
        var moved: Boolean = false
    )

    private val rows = ArrayList<Row>()
    private val filteredRows = ArrayList<Row>()
    private lateinit var adapter: ResultAdapter
    private lateinit var safeTab: TabLayout.Tab
    private lateinit var nsfwTab: TabLayout.Tab
    private var selectedTab = TAB_SAFE
    private var threshold = 0.90f

    private val thumbCache = object : LruCache<String, Bitmap>(
        (Runtime.getRuntime().maxMemory() / 1024L / 8L).toInt().coerceAtLeast(4096)
    ) {
        override fun sizeOf(key: String, value: Bitmap): Int = (value.byteCount / 1024).coerceAtLeast(1)
    }

    /** 加载模型 / 识别 / 移动中（此时禁用所有按钮） */
    private var busy = false

    /** 正在扫描文件夹（此时隐藏判定阈值模块） */
    private var scanning = false

    /** 正在下载模型（此时下载按钮作为“停止下载”使用） */
    private var downloading = false

    @Volatile
    private var cancelDownload = false

    /** 当前下载连接，用于立即中断阻塞中的 read */
    @Volatile
    private var activeConn: HttpURLConnection? = null

    // 选择图片文件夹
    private val pickFolder =
        registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { treeUri ->
            if (treeUri != null) scoreFolder(treeUri)
        }

    // 从本地导入 GGUF 模型
    private val importModel =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) copyModel(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.topAppBar.setOnMenuItemClickListener { item ->
            if (item.itemId == R.id.action_language) {
                toggleLanguage()
                true
            } else {
                false
            }
        }

        safeTab = binding.tabCategory.newTab().setText(getString(R.string.tab_safe, 0))
        nsfwTab = binding.tabCategory.newTab().setText(getString(R.string.tab_nsfw, 0))
        binding.tabCategory.addTab(safeTab)
        binding.tabCategory.addTab(nsfwTab)
        binding.tabCategory.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) {
                selectedTab = tab.position
                renderRows()
            }

            override fun onTabUnselected(tab: TabLayout.Tab) {}
            override fun onTabReselected(tab: TabLayout.Tab) {}
        })

        adapter = ResultAdapter()
        binding.listResults.layoutManager = GridLayoutManager(this, 2)
        binding.listResults.adapter = adapter

        binding.btnDownload.setOnClickListener { onDownloadClick() }
        binding.btnImport.setOnClickListener { importModel.launch(arrayOf("*/*")) }
        binding.btnPick.setOnClickListener {
            if (!NsfwEngine.isLoaded) {
                toast(getString(R.string.status_need_model))
                return@setOnClickListener
            }
            pickFolder.launch(null)
        }

        binding.tvThreshold.text = getString(R.string.label_threshold, threshold)
        binding.seekThreshold.addOnChangeListener { _, value, _ ->
            threshold = value
            binding.tvThreshold.text = getString(R.string.label_threshold, threshold)
            renderRows()
        }

        updateModelActions()
        renderRows()

        // 启动时按已下载的模型自动加载（语言切换重建时若已加载则直接复用）
        if (NsfwEngine.isLoaded) {
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)
            status(getString(R.string.status_model_ready, NsfwEngine.backend(), threads))
        } else if (modelFile().exists()) {
            loadEngine()
        } else if (partFile().exists()) {
            status(getString(R.string.status_incomplete_download))
            setProgressIndeterminate(true)
            showProgress()
            binding.tvProgress.text =
                getString(R.string.progress_downloaded_mb, partFile().length() / 1048576)
        } else {
            status(getString(R.string.status_model_missing))
        }
    }

    override fun onDestroy() {
        cancelDownload = true
        try {
            activeConn?.disconnect()
        } catch (_: Throwable) {
        }
        super.onDestroy()
        executor.shutdownNow()
        thumbExecutor.shutdownNow()
    }

    private fun toggleLanguage() {
        if (busy || downloading) return
        val currentLocales = AppCompatDelegate.getApplicationLocales()
        val isChinese = if (!currentLocales.isEmpty) {
            currentLocales.toLanguageTags().startsWith("zh", ignoreCase = true)
        } else {
            resources.configuration.locales[0]?.language?.startsWith("zh", ignoreCase = true) == true
        }
        val target = if (isChinese) "en" else "zh-CN"
        AppCompatDelegate.setApplicationLocales(LocaleListCompat.forLanguageTags(target))
    }

    // ============================ 模型管理 ============================

    private fun modelDir(): File {
        val base = getExternalFilesDir(null) ?: filesDir
        return File(base, "models").apply { mkdirs() }
    }

    private fun modelFile(): File = File(modelDir(), MODEL_FILE)

    private fun partFile(): File = File(modelFile().path + ".part")

    /** 模型已存在时隐藏下载/导入入口，扫描中隐藏判定阈值模块 */
    private fun updateModelActions() {
        val hasModel = modelFile().exists()
        val showActions = !hasModel

        binding.rowModelActions.visibility = if (showActions) View.VISIBLE else View.GONE
        if (!showActions) hideProgress()

        binding.cardThreshold.visibility = if (scanning) View.GONE else View.VISIBLE

        binding.btnDownload.text = when {
            downloading -> getString(R.string.btn_stop_download)
            partFile().exists() -> getString(R.string.btn_resume_download)
            else -> getString(R.string.btn_download)
        }
        binding.btnDownload.isEnabled = !busy
        binding.btnImport.isEnabled = !busy && !downloading
        binding.btnPick.isEnabled = !busy && !downloading
    }

    private fun showProgress() {
        binding.progressBar.visibility = View.VISIBLE
        binding.tvProgress.visibility = View.VISIBLE
    }

    private fun hideProgress() {
        binding.progressBar.visibility = View.GONE
        binding.tvProgress.visibility = View.GONE
    }

    private fun loadEngine() {
        if (busy) return
        val f = modelFile()
        if (!f.exists()) {
            status(getString(R.string.status_need_model))
            return
        }
        setBusy(true, getString(R.string.status_loading_model))
        executor.execute {
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)
            val err = NsfwEngine.load(f.absolutePath, threads)
            main.post {
                setBusy(
                    false,
                    if (err == null) getString(R.string.status_model_ready, NsfwEngine.backend(), threads)
                    else getString(R.string.status_model_load_failed, err)
                )
            }
        }
    }

    /** 云端下载按钮：下载中再次点击则停止 */
    private fun onDownloadClick() {
        if (downloading) {
            cancelDownload = true
            try {
                activeConn?.disconnect()
            } catch (_: Throwable) {
            }
            status(getString(R.string.status_stopping_download))
            return
        }
        if (busy) return

        if (isCellular()) {
            MaterialAlertDialogBuilder(this)
                .setTitle(R.string.dialog_cellular_title)
                .setMessage(R.string.dialog_cellular_message)
                .setPositiveButton(R.string.dialog_continue) { _, _ -> downloadModel() }
                .setNegativeButton(R.string.dialog_cancel, null)
                .show()
        } else {
            downloadModel()
        }
    }

    /**
     * 支持断点续传（HTTP Range）；单次点击内遇到网络中断会自动续传重试，
     * 全部成功后校验 SHA256 再原子重命名。
     */
    private fun downloadModel() {
        if (busy || downloading) return
        val target = modelFile()
        val part = partFile()

        downloading = true
        cancelDownload = false
        updateModelActions()
        status(getString(R.string.status_downloading))
        setProgressIndeterminate(false)
        showProgress()
        binding.progressBar.progress = 0
        binding.tvProgress.text = getString(R.string.progress_connecting)

        executor.execute {
            var attempt = 0
            var finished = false
            while (attempt < MAX_RETRY && !cancelDownload) {
                attempt++
                try {
                    fetchTo(part)
                    if (cancelDownload) break
                    finished = true
                    break
                } catch (t: Throwable) {
                    if (cancelDownload) break
                    if (attempt >= MAX_RETRY) {
                        main.post {
                            downloading = false
                            activeConn = null
                            updateModelActions()
                            hideProgress()
                            status(getString(R.string.status_download_failed, t.message ?: ""))
                        }
                        return@execute
                    }
                    main.post {
                        status(getString(R.string.status_retrying, attempt, MAX_RETRY))
                    }
                    try {
                        Thread.sleep(1500)
                    } catch (_: InterruptedException) {
                        break
                    }
                }
            }

            if (!finished) {
                // 用户停止：保留 .part，下次点击可续传
                main.post {
                    downloading = false
                    activeConn = null
                    updateModelActions()
                    status(getString(R.string.status_download_paused))
                    binding.tvProgress.text =
                        getString(R.string.progress_paused_mb, part.length() / 1048576)
                }
                return@execute
            }

            try {
                if (part.length() < MODEL_MIN_BYTES) {
                    throw IllegalStateException(getString(R.string.error_incomplete_file))
                }
                main.post {
                    status(getString(R.string.status_verifying_model))
                    binding.tvProgress.text = getString(R.string.progress_verifying_sha256)
                    setProgressIndeterminate(true)
                }
                val hash = sha256(part)
                if (!hash.equals(MODEL_SHA256, true)) {
                    part.delete()  // 内容损坏，丢弃后重新下载
                    throw IllegalStateException(getString(R.string.error_sha256_mismatch))
                }
                if (target.exists()) target.delete()
                if (!part.renameTo(target)) {
                    throw IllegalStateException(getString(R.string.error_save_failed))
                }

                main.post {
                    downloading = false
                    hideProgress()
                    updateModelActions()
                    loadEngine()
                }
            } catch (t: Throwable) {
                main.post {
                    downloading = false
                    activeConn = null
                    updateModelActions()
                    hideProgress()
                    status(getString(R.string.status_download_failed, t.message ?: ""))
                }
            }
        }
    }

    /** 单次下载请求：从 .part 已有长度续传 */
    private fun fetchTo(part: File) {
        var resumeFrom = if (part.exists()) part.length() else 0L
        val conn = (URL(MODEL_URL).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20000
            readTimeout = 30000
            if (resumeFrom > 0) setRequestProperty("Range", "bytes=$resumeFrom-")
        }
        activeConn = conn
        conn.connect()
        val code = conn.responseCode
        if (code == 200) {
            // 服务器不支持断点续传，从头开始
            resumeFrom = 0
            part.delete()
        } else if (code != 206) {
            throw IllegalStateException("HTTP $code")
        }
        val total = if (code == 206) resumeFrom + conn.contentLength
        else conn.contentLength.toLong()

        var got = 0L
        var lastPct = -1
        conn.inputStream.use { input ->
            FileOutputStream(part, resumeFrom > 0).use { out ->
                val buf = ByteArray(256 * 1024)
                while (true) {
                    if (cancelDownload) break
                    val n = input.read(buf)
                    if (n < 0) break
                    out.write(buf, 0, n)
                    got += n
                    val pct = if (total > 0) {
                        (((resumeFrom + got) * 100) / total).toInt()
                    } else -1
                    if (pct != lastPct) {
                        lastPct = pct
                        publishProgress(pct, resumeFrom + got, total)
                    }
                }
            }
        }
        conn.disconnect()
        activeConn = null

        if (cancelDownload) return
        if (total > 0 && resumeFrom + got < total) {
            throw IllegalStateException(getString(R.string.error_connection_interrupted))
        }
    }

    private fun publishProgress(pct: Int, got: Long, total: Long) {
        main.post {
            val gotMb = got / 1048576
            val totalMb = total / 1048576
            if (pct >= 0) {
                setProgressIndeterminate(false)
                binding.progressBar.progress = pct
                binding.tvProgress.text =
                    getString(R.string.progress_percent_mb, pct, gotMb, totalMb)
            } else {
                setProgressIndeterminate(true)
                binding.tvProgress.text = getString(R.string.progress_mb_only, gotMb)
            }
        }
    }

    /** 本地导入：只接受 .gguf 文件，并校验 GGUF 文件头 */
    private fun copyModel(uri: Uri) {
        if (busy) return
        if (!displayName(uri).lowercase().endsWith(".gguf")) {
            toast(getString(R.string.toast_only_gguf))
            return
        }

        setBusy(true, getString(R.string.status_importing_model))
        executor.execute {
            val target = modelFile()
            val part = partFile()
            try {
                val input = contentResolver.openInputStream(uri)
                    ?: throw IllegalStateException(getString(R.string.error_cannot_read_file))
                input.use { src ->
                    // 校验文件头 magic = "GGUF"
                    val magic = ByteArray(4)
                    var read = 0
                    while (read < 4) {
                        val n = src.read(magic, read, 4 - read)
                        if (n <= 0) break
                        read += n
                    }
                    if (read < 4 || String(magic, Charsets.US_ASCII) != "GGUF") {
                        throw IllegalStateException(getString(R.string.error_not_gguf))
                    }
                    FileOutputStream(part).use { out ->
                        out.write(magic, 0, 4)
                        src.copyTo(out, 256 * 1024)
                    }
                }
                if (part.length() < MODEL_MIN_BYTES) {
                    throw IllegalStateException(getString(R.string.error_file_too_small))
                }
                main.post { status(getString(R.string.status_verifying_model)) }
                val hash = sha256(part)
                if (!hash.equals(MODEL_SHA256, true)) {
                    throw IllegalStateException(getString(R.string.error_import_sha256_mismatch))
                }
                if (target.exists()) target.delete()
                if (!part.renameTo(target)) {
                    throw IllegalStateException(getString(R.string.error_save_failed))
                }
                main.post {
                    setBusy(false, null)
                    loadEngine()
                }
            } catch (t: Throwable) {
                part.delete()
                main.post {
                    setBusy(false, getString(R.string.status_import_failed, t.message ?: ""))
                }
            }
        }
    }

    // ============================ 图片识别 ============================

    private fun scoreFolder(treeUri: Uri) {
        if (!NsfwEngine.isLoaded) {
            toast(getString(R.string.status_need_model))
            return
        }
        if (busy) return

        rows.clear()
        renderRows()
        scanning = true
        setBusy(true, getString(R.string.status_scanning_folder))

        executor.execute {
            val items = collectImages(treeUri)
            if (items.isEmpty()) {
                main.post {
                    scanning = false
                    setBusy(false, getString(R.string.status_no_images_found))
                    toast(getString(R.string.status_no_images_found))
                }
                return@execute
            }

            main.post { status(getString(R.string.status_scoring_progress, 0, items.size, "")) }
            items.forEachIndexed { index, (uri, name) ->
                val bytes = try {
                    contentResolver.openInputStream(uri)?.use { it.readBytes() }
                } catch (_: Throwable) {
                    null
                }
                val p = if (bytes != null) NsfwEngine.score(bytes) else Float.NaN
                main.post {
                    rows.add(Row(uri, name, p))
                    renderRows()
                    status(getString(R.string.status_scoring_progress, index + 1, items.size, name))
                }
            }
            main.post {
                scanning = false
                setBusy(false, getString(R.string.status_scoring_done, items.size))
                val nsfwToMove = rows.filter { !it.moved && !it.score.isNaN() && it.score >= threshold }
                if (nsfwToMove.isNotEmpty()) {
                    promptMoveNsfw(nsfwToMove)
                }
            }
        }
    }

    private fun promptMoveNsfw(targets: List<Row>) {
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.dialog_move_title)
            .setMessage(getString(R.string.dialog_move_message, targets.size))
            .setPositiveButton(R.string.dialog_move_confirm) { _, _ -> moveNsfwToInternal(targets) }
            .setNegativeButton(R.string.dialog_cancel, null)
            .show()
    }

    private fun moveNsfwToInternal(targets: List<Row>) {
        if (busy) return
        val destDir = File(filesDir, "nsfw").apply { mkdirs() }
        setBusy(true, getString(R.string.status_moving_nsfw, 0, targets.size))

        executor.execute {
            var moved = 0
            var failed = 0
            targets.forEachIndexed { index, row ->
                val srcUri = row.uri
                val fileName = row.name.substringAfterLast('/')
                val dstFile = uniqueDestFile(destDir, fileName)
                val ok = try {
                    val copied = contentResolver.openInputStream(srcUri)?.use { input ->
                        FileOutputStream(dstFile).use { output ->
                            input.copyTo(output, 256 * 1024)
                            output.flush()
                        }
                        true
                    } ?: false
                    if (copied && DocumentsContract.deleteDocument(contentResolver, srcUri)) {
                        true
                    } else {
                        dstFile.delete()
                        false
                    }
                } catch (_: Throwable) {
                    dstFile.delete()
                    false
                }

                if (ok) {
                    val newUri = Uri.fromFile(dstFile)
                    val cached = thumbCache.get(srcUri.toString())
                    if (cached != null) thumbCache.put(newUri.toString(), cached)
                    main.post {
                        row.uri = newUri
                        row.moved = true
                    }
                    moved++
                } else {
                    failed++
                }

                main.post {
                    status(getString(R.string.status_moving_nsfw, index + 1, targets.size))
                }
            }

            main.post {
                renderRows()
                val msg = if (failed == 0) {
                    getString(R.string.status_move_done, moved)
                } else {
                    getString(R.string.status_move_partial, moved, failed)
                }
                setBusy(false, msg)
                toast(msg)
            }
        }
    }

    private fun uniqueDestFile(dir: File, fileName: String): File {
        var candidate = File(dir, fileName)
        if (!candidate.exists()) return candidate
        val dot = fileName.lastIndexOf('.')
        val base = if (dot > 0) fileName.substring(0, dot) else fileName
        val ext = if (dot > 0) fileName.substring(dot) else ""
        var i = 1
        while (candidate.exists()) {
            candidate = File(dir, "$base ($i)$ext")
            i++
        }
        return candidate
    }

    private fun collectImages(treeUri: Uri): List<Pair<Uri, String>> {
        val result = ArrayList<Pair<Uri, String>>()
        val rootDocId = DocumentsContract.getTreeDocumentId(treeUri) ?: return result
        val queue = ArrayDeque<Pair<String, String>>()
        val visited = HashSet<String>()
        queue.add(rootDocId to "")

        val projection = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE
        )

        while (queue.isNotEmpty()) {
            val (docId, prefix) = queue.removeFirst()
            if (!visited.add(docId)) continue

            val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, docId)
            try {
                contentResolver.query(childrenUri, projection, null, null, null)?.use { c ->
                    val idIdx = c.getColumnIndex(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
                    val nameIdx = c.getColumnIndex(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
                    val mimeIdx = c.getColumnIndex(DocumentsContract.Document.COLUMN_MIME_TYPE)
                    if (idIdx < 0) return@use

                    while (c.moveToNext()) {
                        val childId = c.getString(idIdx) ?: continue
                        val name = if (nameIdx >= 0) c.getString(nameIdx) ?: "image" else "image"
                        if (name.startsWith(".")) continue

                        val mime = if (mimeIdx >= 0) c.getString(mimeIdx) else null
                        val displayPath = if (prefix.isEmpty()) name else "$prefix/$name"
                        if (mime == DocumentsContract.Document.MIME_TYPE_DIR) {
                            queue.add(childId to displayPath)
                        } else if (isImageFile(name, mime)) {
                            val fileUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, childId)
                            result.add(fileUri to displayPath)
                        }
                    }
                }
            } catch (_: Throwable) {
            }
        }
        result.sortBy { it.second.lowercase() }
        return result
    }

    private fun isImageFile(name: String, mimeType: String?): Boolean {
        if (mimeType != null && mimeType.startsWith("image/", ignoreCase = true)) return true
        val ext = name.substringAfterLast('.', "").lowercase()
        return ext in IMAGE_EXTENSIONS
    }

    private fun renderRows() {
        val safeList = ArrayList<Row>()
        val nsfwList = ArrayList<Row>()
        rows.forEach { row ->
            val p = row.score
            if (!p.isNaN() && p >= threshold) {
                nsfwList.add(row)
            } else {
                safeList.add(row)
            }
        }

        safeTab.text = getString(R.string.tab_safe, safeList.size)
        nsfwTab.text = getString(R.string.tab_nsfw, nsfwList.size)

        filteredRows.clear()
        filteredRows.addAll(if (selectedTab == TAB_NSFW) nsfwList else safeList)
        adapter.notifyDataSetChanged()

        val hasAny = rows.isNotEmpty()
        binding.tvSummary.visibility = if (hasAny) View.VISIBLE else View.GONE
        binding.tvSummary.text = if (!hasAny) "" else
            getString(R.string.summary_format, rows.size, nsfwList.size, threshold)

        if (!hasAny) {
            binding.emptyState.visibility = View.VISIBLE
            binding.tvEmptyTitle.setText(R.string.empty_title)
            binding.tvEmptySubtitle.visibility = View.VISIBLE
            binding.tvEmptySubtitle.setText(R.string.empty_subtitle)
        } else if (filteredRows.isEmpty()) {
            binding.emptyState.visibility = View.VISIBLE
            binding.tvEmptyTitle.setText(R.string.empty_category)
            binding.tvEmptySubtitle.visibility = View.GONE
        } else {
            binding.emptyState.visibility = View.GONE
        }
    }

    private fun setProgressIndeterminate(indeterminate: Boolean) {
        if (binding.progressBar.isIndeterminate == indeterminate) return
        val vis = binding.progressBar.visibility
        if (vis == View.VISIBLE) binding.progressBar.visibility = View.INVISIBLE
        binding.progressBar.isIndeterminate = indeterminate
        binding.progressBar.visibility = vis
    }

    private fun decodeThumbnail(uri: Uri, targetPx: Int = 360): Bitmap? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
            val maxDim = maxOf(bounds.outWidth, bounds.outHeight)
            var sample = 1
            if (maxDim > targetPx) {
                while (maxDim / (sample * 2) >= targetPx) {
                    sample *= 2
                }
            }
            val opts = BitmapFactory.Options().apply {
                inSampleSize = sample
                inPreferredConfig = Bitmap.Config.RGB_565
            }
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
        } catch (_: Throwable) {
            null
        }
    }

    private inner class ResultAdapter : RecyclerView.Adapter<ResultAdapter.Holder>() {
        inner class Holder(val b: ItemResultBinding) : RecyclerView.ViewHolder(b.root)

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder {
            return Holder(ItemResultBinding.inflate(LayoutInflater.from(parent.context), parent, false))
        }

        override fun getItemCount(): Int = filteredRows.size

        override fun onBindViewHolder(holder: Holder, position: Int) {
            val row = filteredRows[position]
            val b = holder.b
            val root = b.root
            b.tvName.text = row.name

            val key = row.uri.toString()
            b.ivThumb.tag = key
            val cached = thumbCache.get(key)
            if (cached != null) {
                b.ivThumb.setImageBitmap(cached)
            } else {
                b.ivThumb.setImageDrawable(null)
                thumbExecutor.execute {
                    val bmp = decodeThumbnail(row.uri)
                    if (bmp != null) {
                        thumbCache.put(key, bmp)
                        main.post {
                            if (b.ivThumb.tag == key) {
                                b.ivThumb.setImageBitmap(bmp)
                            }
                        }
                    }
                }
            }

            val p = row.score
            if (p.isNaN()) {
                b.tvBadge.setText(R.string.badge_failed)
                b.tvScore.text = "—"
                val bg = MaterialColors.getColor(root, com.google.android.material.R.attr.colorSurfaceContainerHighest)
                val fg = MaterialColors.getColor(root, com.google.android.material.R.attr.colorOnSurfaceVariant)
                b.tvBadge.background.mutate().setTint(bg)
                b.tvBadge.setTextColor(fg)
                b.tvScore.setTextColor(fg)
            } else {
                val bad = p >= threshold
                b.tvBadge.setText(
                    when {
                        row.moved -> R.string.badge_moved
                        bad -> R.string.badge_nsfw
                        else -> R.string.badge_safe
                    }
                )
                b.tvScore.text = String.format("%.2f%%", p * 100)
                val bg = MaterialColors.getColor(
                    root,
                    if (bad) com.google.android.material.R.attr.colorErrorContainer
                    else com.google.android.material.R.attr.colorTertiaryContainer
                )
                val fg = MaterialColors.getColor(
                    root,
                    if (bad) com.google.android.material.R.attr.colorOnErrorContainer
                    else com.google.android.material.R.attr.colorOnTertiaryContainer
                )
                val scoreColor = MaterialColors.getColor(
                    root,
                    if (bad) androidx.appcompat.R.attr.colorError
                    else com.google.android.material.R.attr.colorTertiary
                )
                b.tvBadge.background.mutate().setTint(bg)
                b.tvBadge.setTextColor(fg)
                b.tvScore.setTextColor(scoreColor)
            }
        }
    }

    private fun displayName(uri: Uri): String {
        try {
            contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { c ->
                    if (c.moveToFirst()) return c.getString(0) ?: "image"
                }
        } catch (_: Throwable) {
        }
        return uri.lastPathSegment ?: "image"
    }

    // ============================ 工具 ============================

    private fun setBusy(value: Boolean, message: String?) {
        busy = value
        updateModelActions()
        if (message != null) status(message)
    }

    private fun status(text: String) {
        binding.tvStatus.text = text
    }

    private fun toast(text: String) {
        Toast.makeText(this, text, Toast.LENGTH_SHORT).show()
    }

    private fun isCellular(): Boolean {
        val cm = getSystemService(ConnectivityManager::class.java) ?: return false
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        if (!caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)) return false
        return !caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) &&
                !caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)
    }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(1024 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02X".format(it) }
    }

    companion object {
        private const val TAB_SAFE = 0
        private const val TAB_NSFW = 1
        private const val MODEL_FILE = "nsfw_image_detection.fp32.gguf"
        private const val MODEL_URL =
            "https://github.com/xlzhen-940218/NSFWImageDetectionGGUF/releases/download/1.0.0/nsfw_image_detection.fp32.gguf"
        private const val MODEL_SHA256 =
            "8FB7C772B9CD53F0B179D1ABF385ED4009F19E8DC6F01ABA19D8344416F446D1"
        private const val MODEL_MIN_BYTES = 100L * 1048576
        private const val MAX_RETRY = 3
        private val IMAGE_EXTENSIONS = setOf(
            "jpg", "jpeg", "png", "webp", "bmp", "gif", "heic", "heif", "avif"
        )
    }
}

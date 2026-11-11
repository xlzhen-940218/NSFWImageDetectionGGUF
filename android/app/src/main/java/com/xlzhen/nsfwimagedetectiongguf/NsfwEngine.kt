package com.xlzhen.nsfwimagedetectiongguf

/**
 * NSFW 推理引擎的 Kotlin 封装。
 * 同一时刻只应有一个线程调用 [score]；模型加载/释放与打分解耦，
 * 模型文件未就绪时 [score] 返回 Float.NaN。
 */
object NsfwEngine {

    private var handle: Long = 0L

    /** 最近一次失败原因，成功时为上一次的残留信息，仅供提示 */
    @Volatile
    var lastError: String = ""
        private set

    val isLoaded: Boolean
        get() = handle != 0L

    /** 加载模型；返回 null 表示成功，否则为错误信息 */
    @Synchronized
    fun load(modelPath: String, threads: Int): String? {
        release()
        val h = nativeCreate()
        if (h == 0L) {
            lastError = "引擎创建失败"
            return lastError
        }
        val err = nativeLoad(h, modelPath, threads)
        if (err != null) {
            nativeDestroy(h)
            lastError = err
            return err
        }
        handle = h
        return null
    }

    /** 输入图片文件原始字节，返回 nsfw 概率；失败返回 NaN */
    fun score(bytes: ByteArray): Float {
        val h = handle
        if (h == 0L) {
            lastError = "模型尚未加载"
            return Float.NaN
        }
        val p = nativeScore(h, bytes)
        if (p.isNaN()) {
            lastError = nativeLastError() ?: "识别失败"
        }
        return p
    }

    fun backend(): String = if (handle == 0L) "" else (nativeBackend(handle) ?: "")

    @Synchronized
    fun release() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private external fun nativeCreate(): Long

    private external fun nativeLoad(handle: Long, path: String, threads: Int): String?

    private external fun nativeScore(handle: Long, data: ByteArray): Float

    private external fun nativeBackend(handle: Long): String?

    private external fun nativeLastError(): String?

    private external fun nativeDestroy(handle: Long)

    init {
        System.loadLibrary("nsfwimagedetectiongguf")
    }
}

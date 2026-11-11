// JNI 桥接：把 nsfw::Engine 暴露给 Kotlin
#include <jni.h>

#include <android/log.h>

#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include "nsfw_core.h"

#define LOG_TAG "NsfwEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

using clock_ms = std::chrono::steady_clock;

std::mutex g_err_mtx;
std::string g_last_error;

void set_last_error(const std::string & e) {
    std::lock_guard<std::mutex> lock(g_err_mtx);
    g_last_error = e;
}

nsfw::Engine * engine_of(jlong h) {
    return reinterpret_cast<nsfw::Engine *>(h);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_xlzhen_nsfwimagedetectiongguf_NsfwEngine_nativeCreate(JNIEnv *, jclass) {
    set_last_error("");
    return reinterpret_cast<jlong>(new nsfw::Engine());
}

// 返回 null 表示成功，否则为错误信息
extern "C" JNIEXPORT jstring JNICALL
Java_com_xlzhen_nsfwimagedetectiongguf_NsfwEngine_nativeLoad(
        JNIEnv * env, jclass, jlong handle, jstring path, jint threads) {
    nsfw::Engine * e = engine_of(handle);
    if (!e) return env->NewStringUTF("引擎未初始化");

    const char * p = path ? env->GetStringUTFChars(path, nullptr) : nullptr;
    const auto t0 = clock_ms::now();
    std::string err;
    const bool ok = e->load(p ? p : "", (int) threads, err);
    if (p) env->ReleaseStringUTFChars(path, p);

    const long ms = (long) std::chrono::duration_cast<std::chrono::milliseconds>(
            clock_ms::now() - t0).count();
    if (ok) {
        LOGI("model loaded in %ld ms, backend=%s, threads=%d", ms,
             e->backend_name().c_str(), (int) threads);
        return nullptr;
    }
    LOGE("model load failed in %ld ms: %s", ms, err.c_str());
    set_last_error(err.empty() ? "模型加载失败" : err);
    return env->NewStringUTF(g_last_error.c_str());
}

// 返回 nsfw 概率；失败返回 NaN，错误信息用 nativeLastError 获取
extern "C" JNIEXPORT jfloat JNICALL
Java_com_xlzhen_nsfwimagedetectiongguf_NsfwEngine_nativeScore(
        JNIEnv * env, jclass, jlong handle, jbyteArray data) {
    nsfw::Engine * e = engine_of(handle);
    if (!e || !data) {
        set_last_error("引擎未初始化");
        return std::nanf("");
    }

    const jsize n = env->GetArrayLength(data);
    std::vector<uint8_t> buf((size_t) n);
    env->GetByteArrayRegion(data, 0, n, reinterpret_cast<jbyte *>(buf.data()));

    const auto t0 = clock_ms::now();
    float p = 0.0f;
    std::string err;
    if (!e->score_image(buf.data(), buf.size(), p, err)) {
        set_last_error(err.empty() ? "识别失败" : err);
        return std::nanf("");
    }
    const long ms = (long) std::chrono::duration_cast<std::chrono::milliseconds>(
            clock_ms::now() - t0).count();
    LOGI("scored %d bytes in %ld ms, p_nsfw=%.6f", (int) n, ms, (double) p);
    return (jfloat) p;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_xlzhen_nsfwimagedetectiongguf_NsfwEngine_nativeBackend(
        JNIEnv * env, jclass, jlong handle) {
    nsfw::Engine * e = engine_of(handle);
    if (!e) return env->NewStringUTF("");
    return env->NewStringUTF(e->backend_name().c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_xlzhen_nsfwimagedetectiongguf_NsfwEngine_nativeLastError(
        JNIEnv * env, jclass) {
    std::lock_guard<std::mutex> lock(g_err_mtx);
    return env->NewStringUTF(g_last_error.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_xlzhen_nsfwimagedetectiongguf_NsfwEngine_nativeDestroy(
        JNIEnv *, jclass, jlong handle) {
    delete engine_of(handle);
}

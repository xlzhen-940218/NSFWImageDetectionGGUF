// NSFW 图像识别引擎接口（纯 C++，无平台依赖）
// 由 _nsfw_tool/nsfw_audit.cpp 移植，预处理与 ViT-B/16 前向逻辑保持一致，
// 保证与桌面工具的打分对齐。
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nsfw {

class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    // 载入 GGUF 模型并做一次热身推理；失败返回 false 并写入 err
    bool load(const std::string & gguf_path, int n_threads, std::string & err);

    // 输入为图片文件原始字节（jpg/png/webp/bmp/gif），内部用 stb 解码
    bool score_image(const uint8_t * data, size_t size, float & p_nsfw, std::string & err);

    // 模型是否已成功载入
    bool loaded() const;

    // 后端名称，如 "CPU"
    const std::string & backend_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nsfw

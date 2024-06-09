#include "CoreContext.hpp"

#include "BufferStream.hpp"

#include <cassert>

extern "C" {
struct FfiBuf {
    void *data;
    uint32_t len;
    uint32_t cap;

};

__declspec(dllimport) void *create_context();
__declspec(dllimport) void destroy_context(void *ctx);
__declspec(dllimport) void destroy_ffi_buf(FfiBuf buf);

__declspec(dllimport) FfiBuf get_funcs(void *ctx);
__declspec(dllimport) FfiBuf get_nodes(void *ctx);

}

struct Buf {
    FfiBuf ffi_buf;

    explicit Buf(FfiBuf ffi_buf) : ffi_buf(ffi_buf) {}

    ~Buf() {
        destroy_ffi_buf(ffi_buf);
    }

    [[nodiscard]] uint32_t len() const {
        return ffi_buf.len;
    }

    [[nodiscard]] void *data() const {
        return ffi_buf.data;
    }

    [[nodiscard]] std::string to_string() const {
        if (len() == 0) {
            return {};
        }

        return std::string(reinterpret_cast<char *>(data()), len());
    }


    template<class T>
    [[nodiscard]] std::vector<T> read_vec() const {
        if (len() == 0) {
            return {};
        }

        assert(ffi_buf.len % sizeof(T) == 0); // check that the buffer is a multiple of the size of T

        auto len = ffi_buf.len / sizeof(T);
        auto data = static_cast <T *>(ffi_buf.data);

        std::vector<T> result;
        result.reserve(len);
        for (uint32_t i = 0; i < len; i++) {
            result.push_back(data[i]);
        }
        return result;
    }
};

template<>
std::vector<std::string> Buf::read_vec<std::string>() const {
    if (len() == 0) {
        return {};
    }

    BufferStream stream{data(), len()};
    auto len = stream.read<uint32_t>();

    std::vector<std::string> result{};
    result.reserve(len);
    for (uint32_t i = 0; i < len; i++) {
        result.push_back(stream.read<std::string>());
    }

    return result;
}

struct FfiFunc {
    Buf id;
    Buf name;
    Buf category;
    uint32_t behaviour;
    bool output;
    Buf inputs;
    Buf outputs;
    Buf events;
};

struct FfiNode {
    Buf id;
    Buf func_id;
    Buf name;
    bool output;
    bool cache_outputs;
    Buf inputs;
    Buf outputs;
};

Ctx::Ctx() {
    this->ctx = create_context();
}

Ctx::~Ctx() {
    destroy_context(this->ctx);
    this->ctx = nullptr;
}

std::vector<Func> Ctx::get_funcs() const {
    Buf buf = Buf{::get_funcs(this->ctx)};

    auto funcs = buf.read_vec<FfiFunc>();
    std::vector<Func> result;
    result.reserve(funcs.size());

    for (uint32_t i = 0; i < funcs.size(); i++) {
        auto ffi_func = &funcs[i];

        auto events = ffi_func->events.read_vec<std::string>();

        Func func{
                ffi_func->id.to_string(),
                ffi_func->name.to_string(),
                ffi_func->category.to_string(),
                ffi_func->behaviour,
                ffi_func->output,
                {},
                {},
                events,
        };
        result.push_back(func);
    }

    return result;
}

std::vector<Node> Ctx::get_nodes() const {
    Buf buf = Buf{::get_nodes(this->ctx)};

    auto nodes = buf.read_vec<FfiNode>();
    std::vector<Node> result;
    result.reserve(nodes.size());

    for (uint32_t i = 0; i < nodes.size(); i++) {
        auto ffi_node = &nodes[i];
        Node node{
                ffi_node->id.to_string(),
                ffi_node->func_id.to_string(),
                ffi_node->name.to_string(),
                ffi_node->output,
                ffi_node->cache_outputs,
                {},
                {},
        };
        result.push_back(node);
    }

    return result;
}


target("llaisys-device-nvidia-triton")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    -- add CUDA dependencies
    add_includedirs("/usr/local/cuda/include")
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudart")
    add_syslinks("dl")

    add_files("../src/device/nvidia/triton/*.cpp")

    on_install(function (target) end)
target_end()



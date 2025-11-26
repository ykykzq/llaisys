target("llaisys-device-nvidia-cuda")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler", "-fPIC", "-Wno-deprecated-gpu-targets", {force = true})
        add_culdflags("-Xcompiler=-fPIC", {force = true})
    end

    -- add CUDA dependencies
    add_includedirs("/usr/local/cuda/include")
    add_linkdirs("/usr/local/cuda/lib64")
    add_links("cudart")
    add_syslinks("dl")
    
    add_files("../src/device/nvidia/cuda/*.cu")
    
    set_policy("build.cuda.devlink", true)

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia-cuda")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler", "-fPIC", "-Wno-deprecated-gpu-targets", {force = true})
        add_culdflags("-Xcompiler=-fPIC", {force = true})
    end

    add_files("../src/ops/*/nvidia/cuda/*.cu")
    set_policy("build.cuda.devlink", true)

    on_install(function (target) end)
target_end()


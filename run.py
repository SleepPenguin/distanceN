import subprocess


config_command = [
    "cmake",
    "-B",
    "build-release",
    "-DCMAKE_CUDA_ARCHITECTURES=52",
    "-G",
    "Ninja",
]
build_command = ["cmake", "--build", "build-release", "--config", "Release"]
run_command = ["./build-release/main"]


print("config command:", " ".join(config_command))
config_result = subprocess.run(config_command)
if config_result.returncode != 0:
    print("CMake config failed. Exiting...")
    exit(1)
print("build command:", " ".join(build_command))
build_result = subprocess.run(build_command)
if build_result.returncode != 0:
    print("CMake build failed. Exiting...")
    exit(1)

print("=====start run main=====")
subprocess.run(run_command)

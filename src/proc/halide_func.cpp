#include "proc/halide_func.h"
#include <halide_image_io.h>

Halide::Buffer<uint8_t> LoadImage(std::string filename) {
  Halide::Buffer<uint8_t> hImg = Halide::Tools::load_image(filename);
  return hImg;
}

Halide::Target findGpuTarget() {
  // Start with a target suitable for the machine you're running this on.
  Halide::Target target = Halide::get_host_target();

  std::vector<Halide::Target::Feature> features_to_try;
  if (target.os == Halide::Target::Windows) {
    // Try D3D12 first; if that fails, try OpenCL.
    if (sizeof(void *) == 8) {
      // D3D12Compute support is only available on 64-bit systems at present.
      features_to_try.push_back(Halide::Target::D3D12Compute);
    }
    features_to_try.push_back(Halide::Target::OpenCL);
  } else if (target.os == Halide::Target::OSX) {
    // OS X doesn't update its OpenCL drivers, so they tend to be broken.
    // CUDA would also be a fine choice on machines with NVidia GPUs.
    features_to_try.push_back(Halide::Target::Metal);
  } else {
    features_to_try.push_back(Halide::Target::OpenCL);
  }
  // Uncomment the following lines to also try CUDA:
  // features_to_try.push_back(Target::CUDA);

  for (Halide::Target::Feature f : features_to_try) {
    Halide::Target new_target = target.with_feature(f);
    if (host_supports_target_device(new_target)) {
      return new_target;
    }
  }

  //   printf("Requested GPU(s) are not supported. (Do you have the proper
  //   hardware "
  //          "and/or driver installed?)\n");
  return target;
}
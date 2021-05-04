#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4541)
#endif

#ifndef _WIN32
#define dynamic_cast static_cast
#endif
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/DlError.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif

#include "SnpeLib.h"
#include "core/common/common.h"

#include <iostream>
#include <unordered_map>
#include <memory>
#include "core/common/logging/macros.h"
#include "core/common/logging/logging.h"

bool SnpeLib::IsSnpeAvailable() {
  // fallback cpu should always be available:
  zdl::DlSystem::Runtime_t runtime = { zdl::DlSystem::Runtime_t::CPU_FLOAT32 };
  return zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime);
}

static std::string s_getRuntimeString(const zdl::DlSystem::Runtime_t& t) {
  std::unordered_map<zdl::DlSystem::Runtime_t, std::string> s_names;
  s_names[zdl::DlSystem::Runtime_t::AIP_FIXED8_TF] = "AIP_FIXED8_TF";
  s_names[zdl::DlSystem::Runtime_t::DSP_FIXED8_TF] = "DSP_FIXED8_TF";
  s_names[zdl::DlSystem::Runtime_t::GPU_FLOAT16] = "GPU_FLOAT16";
  s_names[zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID] = "GPU_FLOAT32_16_HYBRID";
  s_names[zdl::DlSystem::Runtime_t::CPU_FLOAT32] = "CPU_FLOAT32";
  if (s_names.find(t) != s_names.end()) {
    return s_names[t];
  }
  return "RUNTIME_UNKNOWN";
}

#ifndef _WIN32
#include <sys/system_properties.h>
/* Get device name
    NOTE : these properties can be queried via adb :
adb shell getprop ro.product.manufacturer
adb shell getprop ro.product.model
*/
static void s_device_get_make_and_model(std::string& make, std::string& model) {
  std::vector<char> man(PROP_VALUE_MAX + 1, 0);
  std::vector<char> mod(PROP_VALUE_MAX + 1, 0);
  /* A length 0 value indicates that the property is not defined */
  int man_len = __system_property_get("ro.product.manufacturer", man.data());
  int mod_len = __system_property_get("ro.product.model", mod.data());
  std::string manufacturer(man.data(), man.data() + man_len);
  std::string mmodel(mod.data(), mod.data() + mod_len);
  make = manufacturer;
  model = mmodel;
}

static bool s_device_uses_dsp_only() {
  std::string make, model;
  s_device_get_make_and_model(make, model);
  // enforce DSP only
  if (make == "Microsoft") {
    return true;
  }
  // Epsilon Selfhost LKG
  if (make == "oema0") {
    return true;
  }
  // Zeta EV2
  if (make == "oemc1" && model == "sf c1") {
    return true;
  }
  // Zeta EV1.2
  if (make == "QUALCOMM" && model == "oemc1") {
    return true;
  }
  // OnePlus 7
  if (make == "OnePlus" && model == "GM1903") {
    return true;
  }
  // OnePlus 7T
  if (make == "OnePlus" && model == "HD1903") {
    return true;
  }
  return false;
}

static bool s_device_must_not_use_dsp() {
  std::string make, model;
  s_device_get_make_and_model(make, model);
  // do not use DSP
  // OnePlus 7Pro
  if (make == "OnePlus" && model == "GM1925") {
    return true;
  }
  return false;
}

#else
static bool s_device_uses_dsp_only() {
  return true;
}
static bool s_device_must_not_use_dsp() {
  return false;
}
#endif

static zdl::DlSystem::Runtime_t s_getPreferredRuntime(bool enforce_dsp) {
  zdl::DlSystem::Runtime_t runtimes[] = { zdl::DlSystem::Runtime_t::DSP_FIXED8_TF,
                                          zdl::DlSystem::Runtime_t::AIP_FIXED8_TF,
                                          zdl::DlSystem::Runtime_t::GPU_FLOAT16,
                                          zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID,
                                          zdl::DlSystem::Runtime_t::CPU_FLOAT32 };
  static zdl::DlSystem::Version_t version = zdl::SNPE::SNPEFactory::getLibraryVersion();
  zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
  LOGS_DEFAULT(INFO) << "SNPE Version %s" << version.asString().c_str(); 

  bool ignore_dsp = s_device_must_not_use_dsp() | !enforce_dsp;
  bool ignore_others = s_device_uses_dsp_only() & enforce_dsp;
  int start = ignore_dsp * 2;
  int end = ignore_others ? 2 : sizeof(runtimes) / sizeof(*runtimes);

  if (ignore_others) {
    runtime = zdl::DlSystem::Runtime_t::DSP;
  }
  // start with skipping aip and dsp if specified. 
  for ( int i=start; i<end; ++i ) {
    LOGS_DEFAULT(INFO) << "testing runtime %d" << (int)runtimes[i];
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtimes[i])) {
      runtime = runtimes[i];
      break;
    }
  }
  LOGS_DEFAULT(INFO) << "using runtime %d" << (int)runtime;
  return runtime;
}

std::string getSnpePreferredRuntimeString(bool enforce_dsp) {
  return s_getRuntimeString(s_getPreferredRuntime(enforce_dsp));
}


class SnpeLibImpl : public SnpeLib {
  zdl::DlSystem::Runtime_t runtime_;
public:
  /*! if false, dsp use is not necessary even if requested by given platform. Not used on Windows. */
  SnpeLibImpl(bool enforce_dsp)
     : runtime_(zdl::DlSystem::Runtime_t::CPU) {
#if defined(_WIN32)
     (void)enforce_dsp; // get rid of unused variable warning
#if !defined(_M_ARM64)
     runtime_ = zdl::DlSystem::Runtime_t::CPU;
#else
    if (enforce_dsp) {
      // force DSP on ARM64 WIN32
      runtime_ = zdl::DlSystem::Runtime_t::DSP;
    }
#endif
#else
    // ANDROID
    runtime_ = s_getPreferredRuntime(enforce_dsp);
#endif
    LOGS_DEFAULT(INFO) << "PerceptionCore using runtime %s" << s_getRuntimeString(runtime_);
  }
  ~SnpeLibImpl() override {}

  std::unique_ptr<zdl::SNPE::SNPE> InitializeSnpe(zdl::DlContainer::IDlContainer* container,
                                                  const std::vector<std::string>* output_tensor_names = nullptr,
                                                  const std::vector<std::string>* input_tensor_names = nullptr) {
    zdl::SNPE::SNPEBuilder snpeBuilder(container);

    // use setOutputTensors instead, also try zdl::DlSystem::Runtime_t::AIP_FIXED8_TF
    //return snpeBuilder.setOutputLayers({}).setRuntimeProcessor(zdl::DlSystem::Runtime_t::DSP_FIXED8_TF).build();

    zdl::DlSystem::StringList dl_output_tensor_names = {};
    if ((nullptr != output_tensor_names) && (output_tensor_names->size() != 0)) {
      for (auto layerName : *output_tensor_names) {
        dl_output_tensor_names.append(layerName.c_str());
      }
    }

    std::unique_ptr<zdl::SNPE::SNPE> snpe = snpeBuilder.setOutputTensors(dl_output_tensor_names).setRuntimeProcessor(runtime_).build();

    input_tensor_map.clear();
    input_tensors.clear();
    if ((snpe != nullptr) && (input_tensor_names != nullptr) && (input_tensor_names->size() != 0)) {
      input_tensors.resize(input_tensor_names->size());
      for (size_t i = 0; i < input_tensor_names->size(); ++i) {
        zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> input_shape = snpe->getInputDimensions(input_tensor_names->at(i).c_str());
        if (!input_shape) {
          LOGS_DEFAULT(ERROR) << "Snpe cannot get input shape for input name: " << input_tensor_names->at(i).c_str();
          input_tensor_map.clear();
          input_tensors.clear();
          return nullptr;
        }
        input_tensors[i] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(*input_shape);
        zdl::DlSystem::ITensor* input_tensor = input_tensors[i].get();
        if (!input_tensor) {
          LOGS_DEFAULT(ERROR) << "Snpe cannot create ITensor";
          input_tensor_map.clear();
          input_tensors.clear();
          return nullptr;
        }
        input_tensor_map.add(input_tensor_names->at(i).c_str(), input_tensor);
      }
    }

    return snpe;
  }

  bool Initialize(const char* dlcPath, const std::vector<std::string>* output_layer_names = nullptr,
                  const std::vector<std::string>* input_layer_names = nullptr) {
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(dlcPath));
    if (!container) {
      LOGS_DEFAULT(ERROR) << "ailed open " << dlcPath << " container file";
      return false;
    }

    snpe_ = InitializeSnpe(container.get(), output_layer_names, input_layer_names);
    if (!snpe_)
    {
      LOGS_DEFAULT(ERROR) << "failed to build snpe";
      return false;
    }

    return true;
  }

  bool Initialize(const unsigned char* dlcData, size_t size, const std::vector<std::string>* output_layer_names = nullptr,
                  const std::vector<std::string>* input_layer_names = nullptr) {
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(dlcData, size);
    if (container == nullptr)
    {
      LOGS_DEFAULT(ERROR) << "failed open container buffer";
      return false;
    }

    snpe_ = InitializeSnpe(container.get(), output_layer_names, input_layer_names);
    if (!snpe_)
    {
      LOGS_DEFAULT(ERROR) << "failed to build snpe " << zdl::DlSystem::getLastErrorString();
      return false;
    }

    return true;
  }

  bool GetInputDimensions(int which, std::vector<int>& sizes) override {
    try {
      zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> input_shape;
      if (which != 0) {
        zdl::DlSystem::Optional<zdl::DlSystem::StringList> pnames = snpe_->getInputTensorNames();
        if (!pnames) {
          LOGS_DEFAULT(ERROR) << "Snpe cannot get input names";
          return false;
        }
        const zdl::DlSystem::StringList& names(*pnames);
        if (names.size() <= which) {
          LOGS_DEFAULT(ERROR) << "Snpe cannot find input " << which;
          return false;
        }
        input_shape = snpe_->getInputDimensions(names.at(which));
      } else {
        input_shape = snpe_->getInputDimensions();
      }
      if (!input_shape) {
        LOGS_DEFAULT(ERROR) << "Snpe cannot get input shape for input " << which;
        return false;
      }
      zdl::DlSystem::TensorShape shape(*input_shape);
      sizes.resize(shape.rank());
      for (size_t i = 0; i < shape.rank(); ++i) {
        sizes[i] = (int)shape[i];  // todo: sizes should be of type size_t
      }
    } catch (...) {
      LOGS_DEFAULT(ERROR) << "Snpe threw exception";
      return false;
    }
    return true;
  }

  bool GetOutputDimensions(int which, std::vector<int>& sizes) override
  {
    (void)which; (void)sizes;

    /*
    zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> inputShape = _snpe->getOutputDimensions();
    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
    zdl::DlSystem::ITensor* tensor = outputTensorMap.getTensor(tensorNames.at(tensorNames.size() - 1));
    */

    return true;
  }

  bool SnpeProcessMultipleOutput(const unsigned char* input, size_t input_size, size_t output_number, unsigned char* outputs[], size_t output_sizes[]) override {
    try {
      zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> inputShape = snpe_->getInputDimensions();
      if (!inputShape) {
        LOGS_DEFAULT(ERROR) << "Snpe cannot get input shape";
        return false;
      }
      std::unique_ptr<zdl::DlSystem::ITensor> input_tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(*inputShape);
      //std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(*inputShape, input, inputSize);
      if (!input_tensor) {
        LOGS_DEFAULT(ERROR) << "Snpe cannot create ITensor";
        return false;
      }
      // ensure size of the input buffer matches input shape buffer size
      if (input_tensor->getSize() * sizeof(float) != input_size) {
        LOGS_DEFAULT(ERROR) << "Snpe input size incorrect: expected " << input_tensor->getSize() * sizeof(float) << "given " << input_size << " bytes";
        return false;
      }
      memcpy(input_tensor->begin().dataPointer(), input, input_size);

      zdl::DlSystem::TensorMap output_tensor_map;
      bool result = snpe_->execute(input_tensor.get(), output_tensor_map);
      if (!result) {
        LOGS_DEFAULT(ERROR) << "Snpe Error while executing the network.";
        return false;
      }
      if (output_tensor_map.size() == 0) {
        return false;
      }

      zdl::DlSystem::StringList tensor_names = output_tensor_map.getTensorNames();

      for (size_t i=0; i < output_number; i++) {
        zdl::DlSystem::ITensor* tensor = output_tensor_map.getTensor(tensor_names.at(i));
        // ensure size of the output buffer matches output shape buffer size
        if (tensor->getSize() * sizeof(float) > output_sizes[i]) {
          LOGS_DEFAULT(ERROR) << "Snpe output size incorrect: output_layer: " << tensor_names.at(i) << " expected "
                              << tensor->getSize() * sizeof(float) << " given " << output_sizes[i] << " bytes.";
          return false;
        }
        memcpy(outputs[i], tensor->cbegin().dataPointer(), tensor->getSize() * sizeof(float));
      }

      return true;
    }
    catch (...){
      LOGS_DEFAULT(ERROR) << "Snpe threw exception";
      return false;
    }
  }


  bool SnpeProcess(const unsigned char* input, size_t input_size, unsigned char* output, size_t output_size) override {
    // Use SnpeProcessMultipleOutput with 1 output layer
    const int output_layer = 1;
    unsigned char* outputs_array[output_layer];
    size_t output_sizes_array[output_layer];
    outputs_array[0] = output;
    output_sizes_array[0] = output_size;
    return SnpeProcessMultipleOutput(input, input_size, output_layer, outputs_array, output_sizes_array);
  }


  bool SnpeProcessMultipleInputsMultipleOutputs(const unsigned char** inputs, const size_t* input_sizes, size_t input_number,
                                                unsigned char** outputs, const size_t* output_sizes, size_t output_number) override {
    try {
      if (input_number != input_tensors.size()) {
        LOGS_DEFAULT(ERROR) << "Snpe number of inputs doesn't match";
        return false;
      }
      for (size_t i=0; i < input_number; ++i) {
        zdl::DlSystem::ITensor* inputTensor = input_tensors[i].get();
        // ensure size of the input buffer matches input shape buffer size
        if (inputTensor->getSize() * 4 != input_sizes[i]) {
          LOGS_DEFAULT(ERROR) << "Snpe input size incorrect: expected %d, given %d bytes" << inputTensor->getSize() * 4 << input_sizes[i];
          return false;
        }
        memcpy(inputTensor->begin().dataPointer(), inputs[i], input_sizes[i]);
      }
      zdl::DlSystem::TensorMap output_tensor_map;
      bool result = snpe_->execute(input_tensor_map, output_tensor_map);
      if (!result) {
        LOGS_DEFAULT(ERROR) << "Snpe Error while executing the network.";
        return false;
      }
      if (output_tensor_map.size() == 0) {
        return false;
      }

      zdl::DlSystem::StringList tensor_names = output_tensor_map.getTensorNames();

      for (size_t i=0; i < output_number; i++)
      {
        zdl::DlSystem::ITensor* tensor = output_tensor_map.getTensor(tensor_names.at(i));
        // ensure size of the output buffer matches output shape buffer size
        if (tensor->getSize() * sizeof(float) > output_sizes[i]) {
          LOGS_DEFAULT(ERROR) << "Snpe output size incorrect: output_layer" << tensor_names.at(i) << " expected "
                              << tensor->getSize() * 4 << " given " << output_sizes[i] << " bytes";
          return false;
        }
        memcpy(outputs[i], tensor->cbegin().dataPointer(), tensor->getSize() * sizeof(float));
      }

      return true;
    }
    catch (...){
      LOGS_DEFAULT(ERROR) << "Snpe threw exception";
      return false;
    }
  }

private:
  std::unique_ptr<zdl::SNPE::SNPE> snpe_;
  std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> input_tensors;
  zdl::DlSystem::TensorMap input_tensor_map;
};

std::unique_ptr<SnpeLib> SnpeLib::SnpeLibFactory(const char* dlc_path, const std::vector<std::string>* output_layer_names,
                                                 bool enforce_dsp, const std::vector<std::string>* input_layer_names) {
  std::unique_ptr<SnpeLibImpl> object(new SnpeLibImpl(enforce_dsp));

  if (!object) {
    ORT_THROW("failed to make snpe library");
  }

  if (!object->Initialize(dlc_path, output_layer_names, input_layer_names)) {
    ORT_THROW("failed to initialize dlc from path");
  }

  return object;
}

std::unique_ptr<SnpeLib> SnpeLib::SnpeLibFactory(const unsigned char* dlc_cata, size_t size,
                                                 const std::vector<std::string>* output_layer_names,
                                                 bool enforce_dsp,
                                                 const std::vector<std::string>* input_layer_names) {
  std::unique_ptr<SnpeLibImpl> object(new SnpeLibImpl(enforce_dsp));

  if (!object) {
    ORT_THROW("failed to make snpe library");
  }

  if (!object->Initialize(dlc_cata, size, output_layer_names, input_layer_names)) {
    ORT_THROW("failed to initialize dlc from buffer");
  }

  return object;
}

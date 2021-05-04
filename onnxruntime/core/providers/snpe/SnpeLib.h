#pragma once

#include <memory>
#include <string>
#include <vector>

class SnpeLib {
public:
  virtual ~SnpeLib() {}
  
  /*!
   * function to see if SNPE is available without need for any network data
   * @return true if SNPE is supported on at least CPU
   */
  static bool IsSnpeAvailable();
  
  /*!
   * function to see if get the preferred runtime on the given platform
   * @return "RUNTIME_UNKNOWN" if unsupported or not handled by linked SNPE API. Other strings
   *   according to snpe docs
   */
  static std::string GetSnpePreferredRuntimeString(bool enforce_dsp = true);
  
  static std::unique_ptr<SnpeLib> SnpeLibFactory(const char* dlc_path, const std::vector<std::string>* output_layer_names = nullptr,
                                                 bool enforce_dsp = true, const std::vector<std::string>* input_layer_names = nullptr);
  static std::unique_ptr<SnpeLib> SnpeLibFactory(const unsigned char* dlc_data, size_t size, const std::vector<std::string>* output_layer_names = nullptr,
                                                 bool enforce_dsp = true, const std::vector<std::string>* input_layer_names = nullptr);
  
  virtual bool SnpeProcess(const unsigned char* input, size_t input_size, unsigned char* output, size_t output_size) = 0;
  virtual bool SnpeProcessMultipleOutput(const unsigned char* input, size_t input_size, size_t output_number, unsigned char* outputs[], size_t output_sizes[]) = 0;
  virtual bool SnpeProcessMultipleInputsMultipleOutputs(const unsigned char** inputs, const size_t* input_sizes, size_t input_number,
                                                        unsigned char** outputs, const size_t* output_sizes, size_t output_number) = 0;
  
  virtual bool GetInputDimensions(int which, std::vector<int>& shape) = 0;
  virtual bool GetOutputDimensions(int which, std::vector<int>& shape) = 0;
};

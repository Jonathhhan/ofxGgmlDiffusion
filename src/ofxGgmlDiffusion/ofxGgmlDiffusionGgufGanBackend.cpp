#include "ofxGgmlDiffusionGgufGanBackend.h"

#include "ofxGgmlDiffusionUtils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#if __has_include("gguf.h")
#include "gguf.h"
#define OFXGGMLDIFFUSION_HAS_GGUF 1
#else
#define OFXGGMLDIFFUSION_HAS_GGUF 0
#endif

namespace {
	constexpr int pixelLatentSize = 100;
	constexpr int pixelHiddenChannels = 256;
	constexpr int pixelStage0Size = 6;
	constexpr int pixelStage1Channels = 128;
	constexpr int pixelStage2Channels = 64;
	constexpr int pixelOutputChannels = 4;
	constexpr int pixelOutputSize = 24;
	constexpr int pixelKernelSize = 5;
	constexpr float batchNormEpsilon = 0.00001f;
	constexpr float leakyReluSlope = 0.2f;

	ofxGgmlDiffusionResult makeError(const std::string& message) {
		ofxGgmlDiffusionResult result;
		result.success = false;
		result.error = message;
		return result;
	}

	ofxGgmlDiffusionResult makeOk(const std::string& message = "") {
		ofxGgmlDiffusionResult result;
		result.success = true;
		result.text = message;
		return result;
	}

	std::uint64_t mix64(std::uint64_t x) {
		x += 0x9e3779b97f4a7c15ULL;
		x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
		x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
		return x ^ (x >> 31);
	}

	std::uint64_t hashString(const std::string& value) {
		std::uint64_t hash = 1469598103934665603ULL;
		for (unsigned char ch : value) {
			hash ^= static_cast<std::uint64_t>(ch);
			hash *= 1099511628211ULL;
		}
		return hash;
	}

	float splitmix01(std::uint64_t& state) {
		state = mix64(state);
		return static_cast<float>(state >> 40) * (1.0f / 17592186044416.0f); // 2^44
	}

	std::string toLowerCopy(std::string value) {
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
			return static_cast<char>(std::tolower(ch));
		});
		return value;
	}

	bool hasGgufExtension(const std::string& value) {
		if (value.size() < 5) {
			return false;
		}
		return toLowerCopy(value).rfind(".gguf") == value.size() - 5;
	}

	std::string unsupportedModelMessage(const ofxGgmlModelInfo& info) {
		std::string architecture = info.architecture.empty() ? "unknown" : info.architecture;
		return "unsupported GGUF GAN model architecture: " + architecture +
			". This backend currently supports gguf-org/pixel DCGAN checkpoints.";
	}

	float leakyRelu(float value) {
		return value >= 0.0f ? value : value * leakyReluSlope;
	}

	std::size_t chwIndex(int channels, int height, int channel, int y, int x) {
		return (static_cast<std::size_t>(channel) * static_cast<std::size_t>(height) *
				   static_cast<std::size_t>(height)) +
			(static_cast<std::size_t>(y) * static_cast<std::size_t>(height)) +
			static_cast<std::size_t>(x);
	}

	std::size_t convTransposeWeightIndex(
		int outputChannels,
		int inChannel,
		int outChannel,
		int kernelY,
		int kernelX) {
		return (((static_cast<std::size_t>(inChannel) * static_cast<std::size_t>(outputChannels)) +
				   static_cast<std::size_t>(outChannel)) *
				  static_cast<std::size_t>(pixelKernelSize) +
				  static_cast<std::size_t>(kernelY)) *
				static_cast<std::size_t>(pixelKernelSize) +
			static_cast<std::size_t>(kernelX);
	}

	void applyBatchNorm(
		std::vector<float>& values,
		int channels,
		int spatialSize,
		const std::vector<float>& bias,
		const std::vector<float>& mean,
		const std::vector<float>& variance,
		const std::vector<float>& weight) {
		for (int channel = 0; channel < channels; ++channel) {
			const float scale = weight[channel] /
				std::sqrt(std::max(variance[channel], 0.0f) + batchNormEpsilon);
			const float offset = bias[channel] - mean[channel] * scale;
			for (int y = 0; y < spatialSize; ++y) {
				for (int x = 0; x < spatialSize; ++x) {
					const auto index = chwIndex(channels, spatialSize, channel, y, x);
					values[index] = values[index] * scale + offset;
				}
			}
		}
	}

	void applyBatchNormFeatures(
		std::vector<float>& values,
		const std::vector<float>& bias,
		const std::vector<float>& mean,
		const std::vector<float>& variance,
		const std::vector<float>& weight) {
		for (std::size_t i = 0; i < values.size(); ++i) {
			const float scale = weight[i] /
				std::sqrt(std::max(variance[i], 0.0f) + batchNormEpsilon);
			values[i] = values[i] * scale + bias[i] - mean[i] * scale;
		}
	}

	void applyLeakyRelu(std::vector<float>& values) {
		for (auto& value : values) {
			value = leakyRelu(value);
		}
	}

	void convTranspose2d(
		const std::vector<float>& input,
		std::vector<float>& output,
		const std::vector<float>& weights,
		int inputChannels,
		int outputChannels,
		int inputSize,
		int outputSize,
		int stride,
		int padding,
		int outputPadding) {
		(void)outputPadding;
		output.assign(
			static_cast<std::size_t>(outputChannels) *
				static_cast<std::size_t>(outputSize) *
				static_cast<std::size_t>(outputSize),
			0.0f);

		for (int inChannel = 0; inChannel < inputChannels; ++inChannel) {
			for (int y = 0; y < inputSize; ++y) {
				for (int x = 0; x < inputSize; ++x) {
					const float inputValue = input[chwIndex(inputChannels, inputSize, inChannel, y, x)];
					for (int kernelY = 0; kernelY < pixelKernelSize; ++kernelY) {
						const int outputY = y * stride - padding + kernelY;
						if (outputY < 0 || outputY >= outputSize) {
							continue;
						}
						for (int kernelX = 0; kernelX < pixelKernelSize; ++kernelX) {
							const int outputX = x * stride - padding + kernelX;
							if (outputX < 0 || outputX >= outputSize) {
								continue;
							}
							for (int outChannel = 0; outChannel < outputChannels; ++outChannel) {
								const auto outputIndex =
									chwIndex(outputChannels, outputSize, outChannel, outputY, outputX);
								const auto weightIndex = convTransposeWeightIndex(
									outputChannels,
									inChannel,
									outChannel,
									kernelY,
									kernelX);
								output[outputIndex] += inputValue * weights[weightIndex];
							}
						}
					}
				}
			}
		}
	}

	float randomNormal(std::uint64_t& state) {
		const float u1 = std::max(splitmix01(state), std::numeric_limits<float>::min());
		const float u2 = splitmix01(state);
		constexpr float twoPi = 6.28318530717958647692f;
		return std::sqrt(-2.0f * std::log(u1)) * std::cos(twoPi * u2);
	}

	void resetLoadedState(
		bool& loaded,
		std::string& loadedGeneratorPath,
		ofxGgmlModelInfo& modelInfo,
		ofxGgmlDiffusionGgufGanRuntimeKind& runtimeKind,
		ofxGgmlDiffusionGgufGanPixelDcganModel& pixelDcganModel) {
		loaded = false;
		loadedGeneratorPath.clear();
		modelInfo = ofxGgmlModelInfo{};
		runtimeKind = ofxGgmlDiffusionGgufGanRuntimeKind::Unknown;
		pixelDcganModel = ofxGgmlDiffusionGgufGanPixelDcganModel{};
	}

#if OFXGGMLDIFFUSION_HAS_GGUF
	bool loadTensorAsFloat(
		const std::string& path,
		const gguf_context* context,
		std::ifstream& file,
		const std::string& name,
		std::size_t expectedCount,
		std::vector<float>& values,
		std::string& error) {
		const int64_t tensorId = gguf_find_tensor(context, name.c_str());
		if (tensorId < 0) {
			error = "GGUF Pixel/DCGAN tensor is missing: " + name;
			return false;
		}

		const auto tensorType = gguf_get_tensor_type(context, tensorId);
		const auto tensorSize = gguf_get_tensor_size(context, tensorId);
		const auto tensorOffset = gguf_get_data_offset(context) + gguf_get_tensor_offset(context, tensorId);
		values.assign(expectedCount, 0.0f);

		file.clear();
		file.seekg(static_cast<std::streamoff>(tensorOffset), std::ios::beg);
		if (!file) {
			error = "could not seek GGUF tensor data for " + name + " in " + path;
			return false;
		}

		if (tensorType == GGML_TYPE_F32) {
			if (tensorSize != expectedCount * sizeof(float)) {
				error = "GGUF Pixel/DCGAN tensor has unexpected F32 size: " + name;
				return false;
			}
			file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(tensorSize));
			if (!file) {
				error = "could not read GGUF F32 tensor: " + name;
				return false;
			}
			return true;
		}

		if (tensorType == GGML_TYPE_F16) {
			if (tensorSize != expectedCount * sizeof(ggml_fp16_t)) {
				error = "GGUF Pixel/DCGAN tensor has unexpected F16 size: " + name;
				return false;
			}
			std::vector<ggml_fp16_t> halfValues(expectedCount);
			file.read(
				reinterpret_cast<char*>(halfValues.data()),
				static_cast<std::streamsize>(tensorSize));
			if (!file) {
				error = "could not read GGUF F16 tensor: " + name;
				return false;
			}
			ggml_fp16_to_fp32_row(
				halfValues.data(),
				values.data(),
				static_cast<int64_t>(expectedCount));
			return true;
		}

		error = "GGUF Pixel/DCGAN tensor must be F16 or F32: " + name;
		return false;
	}

	bool loadPixelDcganModel(
		const std::string& path,
		ofxGgmlDiffusionGgufGanPixelDcganModel& model,
		std::string& error) {
		gguf_init_params params{};
		params.no_alloc = true;
		params.ctx = nullptr;
		gguf_context* context = gguf_init_from_file(path.c_str(), params);
		if (context == nullptr) {
			error = "failed to read GGUF metadata for Pixel/DCGAN: " + path;
			return false;
		}

		std::ifstream file(path, std::ios::binary);
		if (!file) {
			error = "could not open GGUF Pixel/DCGAN file: " + path;
			gguf_free(context);
			return false;
		}

		auto load = [&](const std::string& name, std::size_t expectedCount, std::vector<float>& values) {
			return loadTensorAsFloat(path, context, file, name, expectedCount, values, error);
		};

		const bool ok =
			load("fc.weight", pixelLatentSize * pixelHiddenChannels * pixelStage0Size * pixelStage0Size, model.fcWeight) &&
			load("bn1.bias", pixelHiddenChannels * pixelStage0Size * pixelStage0Size, model.bn1Bias) &&
			load("bn1.running_mean", pixelHiddenChannels * pixelStage0Size * pixelStage0Size, model.bn1Mean) &&
			load("bn1.running_var", pixelHiddenChannels * pixelStage0Size * pixelStage0Size, model.bn1Variance) &&
			load("bn1.weight", pixelHiddenChannels * pixelStage0Size * pixelStage0Size, model.bn1Weight) &&
			load("bn2.bias", pixelStage1Channels, model.bn2Bias) &&
			load("bn2.running_mean", pixelStage1Channels, model.bn2Mean) &&
			load("bn2.running_var", pixelStage1Channels, model.bn2Variance) &&
			load("bn2.weight", pixelStage1Channels, model.bn2Weight) &&
			load("bn3.bias", pixelStage2Channels, model.bn3Bias) &&
			load("bn3.running_mean", pixelStage2Channels, model.bn3Mean) &&
			load("bn3.running_var", pixelStage2Channels, model.bn3Variance) &&
			load("bn3.weight", pixelStage2Channels, model.bn3Weight) &&
			load("conv_transpose1.weight",
				pixelHiddenChannels * pixelStage1Channels * pixelKernelSize * pixelKernelSize,
				model.convTranspose1Weight) &&
			load("conv_transpose2.weight",
				pixelStage1Channels * pixelStage2Channels * pixelKernelSize * pixelKernelSize,
				model.convTranspose2Weight) &&
			load("conv_transpose3.weight",
				pixelStage2Channels * pixelOutputChannels * pixelKernelSize * pixelKernelSize,
				model.convTranspose3Weight);

		gguf_free(context);
		return ok;
	}
#endif

	bool looksLikePixelDcgan(const ofxGgmlModelInfo& info) {
		return info.architecture == "pig" &&
			info.tensorCount >= 19 &&
			info.metadataCount >= 3;
	}

	ofxGgmlDiffusionImage makePixelDcganImage(
		const ofxGgmlDiffusionGgufGanPixelDcganModel& model,
		const ofxGgmlDiffusionRequest& request) {
		const auto cleanPrompt = ofxGgmlDiffusionUtils::cleanPrompt(request.prompt);
		std::uint64_t rngState = hashString(request.gan.generatorPath) ^
			mix64(static_cast<std::uint64_t>(request.seed));
		if (!cleanPrompt.empty()) {
			rngState ^= hashString(cleanPrompt);
		}
		const std::uint64_t styleKey = rngState ^ 0x6d4f2b3a5c8197e1ULL;

		std::array<float, pixelLatentSize> latent{};
		const float truncation = request.gan.truncation > 0.0f ? request.gan.truncation : 1.0f;
		for (auto& value : latent) {
			value = std::clamp(randomNormal(rngState), -truncation, truncation);
		}

		std::vector<float> stage0(pixelHiddenChannels * pixelStage0Size * pixelStage0Size, 0.0f);
		for (int outputIndex = 0; outputIndex < static_cast<int>(stage0.size()); ++outputIndex) {
			float sum = 0.0f;
			for (int latentIndex = 0; latentIndex < pixelLatentSize; ++latentIndex) {
				sum += model.fcWeight[static_cast<std::size_t>(outputIndex) * pixelLatentSize + latentIndex] *
					latent[latentIndex];
			}
			stage0[outputIndex] = sum;
		}
		applyBatchNormFeatures(
			stage0,
			model.bn1Bias,
			model.bn1Mean,
			model.bn1Variance,
			model.bn1Weight);
		applyLeakyRelu(stage0);

		std::vector<float> stage1;
		convTranspose2d(
			stage0,
			stage1,
			model.convTranspose1Weight,
			pixelHiddenChannels,
			pixelStage1Channels,
			pixelStage0Size,
			pixelStage0Size,
			1,
			2,
			0);
		applyBatchNorm(
			stage1,
			pixelStage1Channels,
			pixelStage0Size,
			model.bn2Bias,
			model.bn2Mean,
			model.bn2Variance,
			model.bn2Weight);
		applyLeakyRelu(stage1);

		std::vector<float> stage2;
		convTranspose2d(
			stage1,
			stage2,
			model.convTranspose2Weight,
			pixelStage1Channels,
			pixelStage2Channels,
			pixelStage0Size,
			12,
			2,
			2,
			1);
		applyBatchNorm(
			stage2,
			pixelStage2Channels,
			12,
			model.bn3Bias,
			model.bn3Mean,
			model.bn3Variance,
			model.bn3Weight);
		applyLeakyRelu(stage2);

		std::vector<float> stage3;
		convTranspose2d(
			stage2,
			stage3,
			model.convTranspose3Weight,
			pixelStage2Channels,
			pixelOutputChannels,
			12,
			pixelOutputSize,
			2,
			2,
			1);

		ofxGgmlDiffusionImage image;
		image.width = request.width;
		image.height = request.height;
		image.channels = pixelOutputChannels;
		image.pixels.resize(
			static_cast<std::size_t>(image.width) *
			static_cast<std::size_t>(image.height) *
			static_cast<std::size_t>(image.channels));

		const int shiftX = static_cast<int>(mix64(styleKey ^ 0x11ULL) % 7ULL) - 3;
		const int shiftY = static_cast<int>(mix64(styleKey ^ 0x23ULL) % 7ULL) - 3;
		std::array<float, 3> colorGain{};
		std::array<float, 3> colorOffset{};
		for (int channel = 0; channel < 3; ++channel) {
			const auto gainBits = mix64(styleKey ^ (0x101ULL + static_cast<std::uint64_t>(channel)));
			const auto offsetBits = mix64(styleKey ^ (0x201ULL + static_cast<std::uint64_t>(channel)));
			colorGain[channel] = 0.86f + static_cast<float>(gainBits & 1023ULL) * (0.28f / 1023.0f);
			colorOffset[channel] = -0.035f + static_cast<float>(offsetBits & 1023ULL) * (0.07f / 1023.0f);
		}

		for (int y = 0; y < image.height; ++y) {
			const int baseSourceY = std::clamp(y * pixelOutputSize / image.height, 0, pixelOutputSize - 1);
			const int sourceY = (baseSourceY + shiftY + pixelOutputSize) % pixelOutputSize;
			for (int x = 0; x < image.width; ++x) {
				const int baseSourceX = std::clamp(x * pixelOutputSize / image.width, 0, pixelOutputSize - 1);
				const int sourceX = (baseSourceX + shiftX + pixelOutputSize) % pixelOutputSize;
				const auto outputBase =
					(static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width) +
					 static_cast<std::size_t>(x)) *
					static_cast<std::size_t>(image.channels);
				for (int channel = 0; channel < pixelOutputChannels; ++channel) {
					const auto sourceIndex = chwIndex(
						pixelOutputChannels,
						pixelOutputSize,
						channel,
						sourceY,
						sourceX);
					float value = (std::tanh(stage3[sourceIndex]) + 1.0f) * 0.5f;
					if (channel < 3) {
						const auto noiseBits = mix64(
							styleKey ^
							(static_cast<std::uint64_t>(x) * 0x9e3779b185ebca87ULL) ^
							(static_cast<std::uint64_t>(y) * 0xc2b2ae3d27d4eb4fULL) ^
							static_cast<std::uint64_t>(channel));
						const float noise =
							(static_cast<float>((noiseBits >> 40) & 65535ULL) / 65535.0f - 0.5f) * 0.045f;
						value = value * colorGain[channel] + colorOffset[channel] + noise;
					}
					image.pixels[outputBase + static_cast<std::size_t>(channel)] =
						static_cast<std::uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f);
				}
			}
		}
		return image;
	}
}

std::string ofxGgmlDiffusionGgufGanBackend::getBackendName() const {
	return "ggml-gguf-gan";
}

ofxGgmlDiffusionBackendFamily ofxGgmlDiffusionGgufGanBackend::getBackendFamily() const {
	return ofxGgmlDiffusionBackendFamily::GAN;
}

bool ofxGgmlDiffusionGgufGanBackend::isAvailable() const {
	return true;
}

bool ofxGgmlDiffusionGgufGanBackend::isLoaded() const {
	return loaded;
}

ofxGgmlDiffusionResult ofxGgmlDiffusionGgufGanBackend::setup(
	const ofxGgmlDiffusionContextSettings& settings) {
	const std::string& path = settings.modelPath;
	if (path.empty()) {
		resetLoadedState(loaded, loadedGeneratorPath, modelInfo, runtimeKind, pixelDcganModel);
		return makeError("GGUF GAN setup requires context settings.modelPath to point to an exported generator file");
	}
	if (!hasGgufExtension(path)) {
		resetLoadedState(loaded, loadedGeneratorPath, modelInfo, runtimeKind, pixelDcganModel);
		return makeError("production GAN backend expects a .gguf generator file");
	}
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		resetLoadedState(loaded, loadedGeneratorPath, modelInfo, runtimeKind, pixelDcganModel);
		return makeError("production GAN backend could not open generator path: " + path);
	}
	if (loaded &&
		loadedGeneratorPath == path &&
		runtimeKind == ofxGgmlDiffusionGgufGanRuntimeKind::PixelDcgan) {
		return makeOk("Pixel/DCGAN GGUF model already loaded from " + path);
	}

	ofxGgmlModel model;
	const auto inspectResult = model.inspect(path);
	if (!inspectResult) {
		resetLoadedState(loaded, loadedGeneratorPath, modelInfo, runtimeKind, pixelDcganModel);
		return makeError(inspectResult.error().message);
	}

	modelInfo = inspectResult.value();
	modelInfo.path = path;
	if (!looksLikePixelDcgan(modelInfo)) {
		const auto message = unsupportedModelMessage(modelInfo);
		resetLoadedState(loaded, loadedGeneratorPath, modelInfo, runtimeKind, pixelDcganModel);
		return makeError(message);
	}

#if OFXGGMLDIFFUSION_HAS_GGUF
	std::string loadError;
	if (!loadPixelDcganModel(path, pixelDcganModel, loadError)) {
		resetLoadedState(loaded, loadedGeneratorPath, modelInfo, runtimeKind, pixelDcganModel);
		return makeError(loadError);
	}
	runtimeKind = ofxGgmlDiffusionGgufGanRuntimeKind::PixelDcgan;
	loadedGeneratorPath = path;
	loaded = true;
	return makeOk("Pixel/DCGAN GGUF model loaded from " + path);
#else
	resetLoadedState(loaded, loadedGeneratorPath, modelInfo, runtimeKind, pixelDcganModel);
	return makeError("gguf headers are not installed; run ofxGgmlCore scripts/setup-ggml.ps1");
#endif
}

ofxGgmlDiffusionResult ofxGgmlDiffusionGgufGanBackend::generate(
	const ofxGgmlDiffusionRequest& request) {
	const auto validation = ofxGgmlDiffusionUtils::validate(request);
	if (!validation) {
		return makeError(validation.errors.empty() ? "invalid GAN request" : validation.errors.front());
	}
	if (!loaded) {
		return makeError("production GGUF GAN backend has not been set up");
	}
	if (request.gan.generatorPath != loadedGeneratorPath) {
		return makeError("request generatorPath does not match loaded GGUF model");
	}
	if (!hasGgufExtension(request.gan.generatorPath)) {
		return makeError("production GAN generation expects a .gguf generatorPath");
	}
	if (runtimeKind != ofxGgmlDiffusionGgufGanRuntimeKind::PixelDcgan) {
		return makeError(unsupportedModelMessage(modelInfo));
	}

	auto image = makePixelDcganImage(pixelDcganModel, request);
	if (!image.isAllocated()) {
		return makeError("Pixel/DCGAN GGUF generation produced no image");
	}

	ofxGgmlDiffusionResult result;
	result.success = true;
	result.seed = request.seed;
	result.text = "Pixel/DCGAN GGUF production lane";
	result.outputPath = request.outputPath;
	result.references.push_back(request.gan.generatorPath);
	result.images.push_back(std::move(image));
	return result;
}

#include "ofxGgmlDiffusionTinyGanBackend.h"

#include "ofxGgmlDiffusionUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<ggml.h>)
	#include <ggml.h>
	#if __has_include(<ggml-cpu.h>)
		#include <ggml-cpu.h>
		#define OFXGGMLDIFFUSION_HAS_GGML 1
	#else
		#define OFXGGMLDIFFUSION_HAS_GGML 0
	#endif
#else
	#define OFXGGMLDIFFUSION_HAS_GGML 0
#endif

namespace {
	ofxGgmlDiffusionResult makeTinyGanError(const std::string& message) {
		ofxGgmlDiffusionResult result;
		result.success = false;
		result.error = message;
		return result;
	}

	ofxGgmlDiffusionResult makeTinyGanOk(const std::string& message) {
		ofxGgmlDiffusionResult result;
		result.success = true;
		result.text = message;
		return result;
	}

	std::string joinValidationErrors(const ofxGgmlDiffusionValidationResult& validation) {
		std::ostringstream stream;
		for (std::size_t i = 0; i < validation.errors.size(); ++i) {
			if (i > 0) {
				stream << "; ";
			}
			stream << validation.errors[i];
		}
		return stream.str();
	}

	std::string trim(const std::string& value) {
		const auto first = value.find_first_not_of(" \t\r\n");
		if (first == std::string::npos) {
			return "";
		}
		const auto last = value.find_last_not_of(" \t\r\n");
		return value.substr(first, last - first + 1);
	}

	bool startsWith(const std::string& value, const std::string& prefix) {
		return value.size() >= prefix.size() &&
			value.compare(0, prefix.size(), prefix) == 0;
	}

	int parseInt(const std::map<std::string, std::string>& values, const std::string& key, int fallback) {
		const auto found = values.find(key);
		if (found == values.end()) {
			return fallback;
		}
		return std::stoi(found->second);
	}

	std::uint32_t parseSeed(const std::map<std::string, std::string>& values, const std::string& key, std::uint32_t fallback) {
		const auto found = values.find(key);
		if (found == values.end()) {
			return fallback;
		}
		return static_cast<std::uint32_t>(std::stoul(found->second));
	}

	float parseFloat(const std::map<std::string, std::string>& values, const std::string& key, float fallback) {
		const auto found = values.find(key);
		if (found == values.end()) {
			return fallback;
		}
		return std::stof(found->second);
	}

	float deterministicValue(std::uint32_t seed) {
		seed ^= seed >> 16;
		seed *= 0x7feb352dU;
		seed ^= seed >> 15;
		seed *= 0x846ca68bU;
		seed ^= seed >> 16;
		const auto normalized = static_cast<float>(seed & 0xffffU) / 32767.5f - 1.0f;
		return normalized;
	}

#if OFXGGMLDIFFUSION_HAS_GGML
	void fillTensor(struct ggml_tensor* tensor, std::uint32_t seed, float scale) {
		auto* data = ggml_get_data_f32(tensor);
		const auto count = ggml_nelements(tensor);
		for (int64_t i = 0; i < count; ++i) {
			data[i] = deterministicValue(seed + static_cast<std::uint32_t>(i * 2654435761U)) * scale;
		}
	}

	void copyOutputToImage(
		const struct ggml_tensor* output,
		int width,
		int height,
		ofxGgmlDiffusionImage& image) {
		image.width = width;
		image.height = height;
		image.channels = 3;
		image.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U);

		const auto* values = ggml_get_data_f32(output);
		for (std::size_t i = 0; i < image.pixels.size(); ++i) {
			const float value = std::max(-1.0f, std::min(1.0f, values[i]));
			image.pixels[i] = static_cast<std::uint8_t>((value * 0.5f + 0.5f) * 255.0f);
		}
	}
#endif
}

ofxGgmlDiffusionTinyGanPreset ofxGgmlDiffusionMakeDefaultTinyGanPreset() {
	return ofxGgmlDiffusionTinyGanPreset{};
}

bool ofxGgmlDiffusionLoadTinyGanPreset(
	const std::string& path,
	ofxGgmlDiffusionTinyGanPreset& preset,
	std::string& error) {
	preset = ofxGgmlDiffusionMakeDefaultTinyGanPreset();
	error.clear();

	if (path.empty() || startsWith(path, "builtin:")) {
		return true;
	}

	std::ifstream input(path);
	if (!input) {
		error = "could not open tiny GAN preset: " + path;
		return false;
	}

	std::map<std::string, std::string> values;
	std::string line;
	int lineNumber = 0;
	while (std::getline(input, line)) {
		++lineNumber;
		const auto comment = line.find('#');
		if (comment != std::string::npos) {
			line = line.substr(0, comment);
		}
		line = trim(line);
		if (line.empty()) {
			continue;
		}
		const auto equals = line.find('=');
		if (equals == std::string::npos) {
			error = "invalid tiny GAN preset line " + std::to_string(lineNumber) + ": " + line;
			return false;
		}
		values[trim(line.substr(0, equals))] = trim(line.substr(equals + 1));
	}

	try {
		preset.version = parseInt(values, "version", preset.version);
		const auto architecture = values.find("architecture");
		if (architecture != values.end()) {
			preset.architecture = architecture->second;
		}
		preset.latentSize = parseInt(values, "latentSize", preset.latentSize);
		preset.hiddenSize = parseInt(values, "hiddenSize", preset.hiddenSize);
		preset.w1Seed = parseSeed(values, "w1Seed", preset.w1Seed);
		preset.b1Seed = parseSeed(values, "b1Seed", preset.b1Seed);
		preset.w2Seed = parseSeed(values, "w2Seed", preset.w2Seed);
		preset.b2Seed = parseSeed(values, "b2Seed", preset.b2Seed);
		preset.latentScale = parseFloat(values, "latentScale", preset.latentScale);
		preset.w1Scale = parseFloat(values, "w1Scale", preset.w1Scale);
		preset.b1Scale = parseFloat(values, "b1Scale", preset.b1Scale);
		preset.w2Scale = parseFloat(values, "w2Scale", preset.w2Scale);
		preset.b2Scale = parseFloat(values, "b2Scale", preset.b2Scale);
	} catch (const std::exception& exception) {
		error = "invalid tiny GAN preset value: ";
		error += exception.what();
		return false;
	}

	if (preset.version != 1) {
		error = "unsupported tiny GAN preset version: " + std::to_string(preset.version);
		return false;
	}
	if (preset.architecture != "tiny-mlp") {
		error = "unsupported tiny GAN architecture: " + preset.architecture;
		return false;
	}
	if (preset.latentSize < 8 || preset.latentSize > 1024) {
		error = "tiny GAN latentSize must be between 8 and 1024";
		return false;
	}
	if (preset.hiddenSize < 8 || preset.hiddenSize > 512) {
		error = "tiny GAN hiddenSize must be between 8 and 512";
		return false;
	}
	if (!std::isfinite(preset.latentScale) ||
		!std::isfinite(preset.w1Scale) ||
		!std::isfinite(preset.b1Scale) ||
		!std::isfinite(preset.w2Scale) ||
		!std::isfinite(preset.b2Scale)) {
		error = "tiny GAN scales must be finite";
		return false;
	}

	return true;
}

std::string ofxGgmlDiffusionSerializeTinyGanPreset(
	const ofxGgmlDiffusionTinyGanPreset& preset) {
	std::ostringstream stream;
	stream << "# ofxGgmlDiffusion tiny GAN preset v1\n";
	stream << "version=" << preset.version << "\n";
	stream << "architecture=" << preset.architecture << "\n";
	stream << "latentSize=" << preset.latentSize << "\n";
	stream << "hiddenSize=" << preset.hiddenSize << "\n";
	stream << "w1Seed=" << preset.w1Seed << "\n";
	stream << "b1Seed=" << preset.b1Seed << "\n";
	stream << "w2Seed=" << preset.w2Seed << "\n";
	stream << "b2Seed=" << preset.b2Seed << "\n";
	stream << "latentScale=" << preset.latentScale << "\n";
	stream << "w1Scale=" << preset.w1Scale << "\n";
	stream << "b1Scale=" << preset.b1Scale << "\n";
	stream << "w2Scale=" << preset.w2Scale << "\n";
	stream << "b2Scale=" << preset.b2Scale << "\n";
	return stream.str();
}

std::string ofxGgmlDiffusionTinyGanBackend::getBackendName() const {
	return "tiny-ggml-gan";
}

ofxGgmlDiffusionBackendFamily ofxGgmlDiffusionTinyGanBackend::getBackendFamily() const {
	return ofxGgmlDiffusionBackendFamily::GAN;
}

bool ofxGgmlDiffusionTinyGanBackend::isAvailable() const {
#if OFXGGMLDIFFUSION_HAS_GGML
	return true;
#else
	return false;
#endif
}

bool ofxGgmlDiffusionTinyGanBackend::isLoaded() const {
	return loaded;
}

ofxGgmlDiffusionResult ofxGgmlDiffusionTinyGanBackend::setup(
	const ofxGgmlDiffusionContextSettings& settings) {
#if OFXGGMLDIFFUSION_HAS_GGML
	threads = settings.threads > 0 ? settings.threads : 1;
	loaded = true;
	return makeTinyGanOk("tiny ggml GAN backend ready");
#else
	loaded = false;
	return makeTinyGanError("tiny ggml GAN backend is unavailable because ggml headers were not found. Run ofxGgmlCore scripts\\setup-ggml.bat first.");
#endif
}

ofxGgmlDiffusionResult ofxGgmlDiffusionTinyGanBackend::generate(
	const ofxGgmlDiffusionRequest& request) {
	const auto validation = ofxGgmlDiffusionUtils::validate(request);
	if (!validation) {
		return makeTinyGanError(joinValidationErrors(validation));
	}
	if (request.backendFamily != ofxGgmlDiffusionBackendFamily::GAN) {
		return makeTinyGanError("tiny ggml GAN backend only accepts GAN requests");
	}
	if (!loaded) {
		return makeTinyGanError("tiny ggml GAN backend has not been set up");
	}

#if OFXGGMLDIFFUSION_HAS_GGML
	const int width = request.width;
	const int height = request.height;
	const int channels = 3;
	ofxGgmlDiffusionTinyGanPreset preset;
	std::string presetError;
	if (!ofxGgmlDiffusionLoadTinyGanPreset(request.gan.generatorPath, preset, presetError)) {
		return makeTinyGanError(presetError);
	}
	if (request.gan.generatorPath.empty() || startsWith(request.gan.generatorPath, "builtin:")) {
		preset.latentSize = std::max(8, std::min(1024, request.gan.latentSize));
	}
	const int latentSize = preset.latentSize;
	const int hiddenSize = preset.hiddenSize;
	const int outputSize = width * height * channels;
	const std::int64_t seed = request.seed >= 0 ? request.seed : 1;

	const std::size_t tensorBytes =
		static_cast<std::size_t>(latentSize) * sizeof(float) +
		static_cast<std::size_t>(latentSize) * static_cast<std::size_t>(hiddenSize) * sizeof(float) +
		static_cast<std::size_t>(hiddenSize) * sizeof(float) +
		static_cast<std::size_t>(hiddenSize) * static_cast<std::size_t>(outputSize) * sizeof(float) +
		static_cast<std::size_t>(outputSize) * sizeof(float) * 2U;
	const std::size_t memSize = tensorBytes + 32U * 1024U * 1024U;

	struct ggml_init_params params = {};
	params.mem_size = memSize;
	params.mem_buffer = nullptr;
	params.no_alloc = false;
	struct ggml_context* ctx = ggml_init(params);
	if (!ctx) {
		return makeTinyGanError("ggml_init failed for tiny GAN graph");
	}

	struct ggml_tensor* latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latentSize, 1);
	struct ggml_tensor* w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latentSize, hiddenSize);
	struct ggml_tensor* b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hiddenSize);
	struct ggml_tensor* w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hiddenSize, outputSize);
	struct ggml_tensor* b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, outputSize);

	fillTensor(latent, static_cast<std::uint32_t>(seed), preset.latentScale * request.gan.truncation);
	fillTensor(w1, preset.w1Seed, preset.w1Scale);
	fillTensor(b1, preset.b1Seed, preset.b1Scale);
	fillTensor(w2, preset.w2Seed, preset.w2Scale);
	fillTensor(b2, preset.b2Seed, preset.b2Scale);

	struct ggml_tensor* hidden = ggml_tanh(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w1, latent), b1));
	struct ggml_tensor* output = ggml_tanh(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w2, hidden), b2));

	struct ggml_cgraph* graph = ggml_new_graph(ctx);
	ggml_build_forward_expand(graph, output);
	const auto status = ggml_graph_compute_with_ctx(ctx, graph, threads);
	if (status != GGML_STATUS_SUCCESS) {
		ggml_free(ctx);
		return makeTinyGanError(std::string("ggml graph compute failed: ") + ggml_status_to_string(status));
	}

	ofxGgmlDiffusionResult result;
	result.success = true;
	result.text = "generated by tiny ggml GAN proof";
	result.outputPath = request.outputPath;
	result.seed = seed;
	result.references.push_back(request.gan.generatorPath);
	result.images.resize(1);
	copyOutputToImage(output, width, height, result.images.front());
	ggml_free(ctx);
	return result;
#else
	return makeTinyGanError("tiny ggml GAN backend is unavailable because ggml headers were not found");
#endif
}

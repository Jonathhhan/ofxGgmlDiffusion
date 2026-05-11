#include "ofxGgmlDiffusionNativeBackend.h"

#include "ofxGgmlDiffusionUtils.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <sstream>

#if defined(OFXGGMLDIFFUSION_WITH_STABLE_DIFFUSION) && __has_include(<stable-diffusion.h>)
	#include <stable-diffusion.h>
	#define OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION 1
#else
	#define OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION 0
#endif

namespace {
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

#if OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION
	const char* emptyToNull(const std::string& value) {
		return value.empty() ? nullptr : value.c_str();
	}

	void releaseImage(sd_image_t& image) {
		if (image.data) {
			std::free(image.data);
			image.data = nullptr;
		}
		image.width = 0;
		image.height = 0;
		image.channel = 0;
	}

	void releaseImageArray(sd_image_t* images, int count) {
		if (!images) {
			return;
		}
		const int safeCount = count > 0 ? count : 0;
		for (int i = 0; i < safeCount; ++i) {
			releaseImage(images[i]);
		}
		std::free(images);
	}

	scheduler_t toNativeScheduler(ofxGgmlDiffusionScheduler scheduler) {
		switch (scheduler) {
		case ofxGgmlDiffusionScheduler::LCM: return LCM_SCHEDULER;
		case ofxGgmlDiffusionScheduler::Turbo: return LCM_SCHEDULER;
		case ofxGgmlDiffusionScheduler::FlowMatch: return SIMPLE_SCHEDULER;
		case ofxGgmlDiffusionScheduler::Default:
		case ofxGgmlDiffusionScheduler::Auto:
		default:
			return SCHEDULER_COUNT;
		}
	}

	sd_tiling_params_t makeTilingParams(bool enabled) {
		sd_tiling_params_t tiling{};
		tiling.enabled = enabled;
		tiling.target_overlap = 0.5f;
		return tiling;
	}

	ofxGgmlDiffusionImage copyImage(const sd_image_t& source) {
		ofxGgmlDiffusionImage image;
		image.width = static_cast<int>(source.width);
		image.height = static_cast<int>(source.height);
		image.channels = static_cast<int>(source.channel);
		const std::size_t byteSize =
			static_cast<std::size_t>(image.width) *
			static_cast<std::size_t>(image.height) *
			static_cast<std::size_t>(image.channels);
		if (source.data && byteSize > 0) {
			image.pixels.assign(source.data, source.data + byteSize);
		}
		return image;
	}
#endif
}

struct ofxGgmlDiffusionNativeBackend::Impl {
	ofxGgmlDiffusionContextSettings settings;
#if OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION
	sd_ctx_t* context = nullptr;
#endif

	~Impl() {
		close();
	}

	bool isAvailable() const {
		return OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION != 0;
	}

	bool isLoaded() const {
#if OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION
		return context != nullptr;
#else
		return false;
#endif
	}

	void close() {
#if OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION
		if (context) {
			free_sd_ctx(context);
			context = nullptr;
		}
#endif
	}
};

ofxGgmlDiffusionNativeBackend::ofxGgmlDiffusionNativeBackend()
	: impl(std::make_unique<Impl>()) {
}

ofxGgmlDiffusionNativeBackend::~ofxGgmlDiffusionNativeBackend() = default;
ofxGgmlDiffusionNativeBackend::ofxGgmlDiffusionNativeBackend(ofxGgmlDiffusionNativeBackend&& other) noexcept = default;
ofxGgmlDiffusionNativeBackend& ofxGgmlDiffusionNativeBackend::operator=(ofxGgmlDiffusionNativeBackend&& other) noexcept = default;

bool ofxGgmlDiffusionNativeBackend::isAvailable() const {
	return impl && impl->isAvailable();
}

bool ofxGgmlDiffusionNativeBackend::isLoaded() const {
	return impl && impl->isLoaded();
}

std::string ofxGgmlDiffusionNativeBackend::getBackendName() const {
	return "stable-diffusion.cpp";
}

ofxGgmlDiffusionContextSettings ofxGgmlDiffusionNativeBackend::getSettings() const {
	return impl ? impl->settings : ofxGgmlDiffusionContextSettings{};
}

ofxGgmlDiffusionResult ofxGgmlDiffusionNativeBackend::setup(const ofxGgmlDiffusionContextSettings& settings) {
	if (!impl) {
		return makeError("native diffusion backend is not initialized");
	}
	impl->close();
	impl->settings = settings;

#if OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION
	if (!settings.hasAnyModelPath()) {
		return makeError("no diffusion model path was configured");
	}

	sd_ctx_params_t params{};
	sd_ctx_params_init(&params);
	params.model_path = emptyToNull(settings.modelPath);
	params.diffusion_model_path = emptyToNull(settings.diffusionModelPath);
	params.clip_l_path = emptyToNull(settings.clipLPath);
	params.clip_g_path = emptyToNull(settings.clipGPath);
	params.t5xxl_path = emptyToNull(settings.t5xxlPath);
	params.vae_path = emptyToNull(settings.vaePath);
	params.taesd_path = emptyToNull(settings.taesdPath);
	params.n_threads = settings.threads;
	params.enable_mmap = settings.mmap;
	params.flash_attn = settings.flashAttention;
	params.diffusion_flash_attn = settings.flashAttention;

	impl->context = new_sd_ctx(&params);
	if (!impl->context) {
		return makeError("stable-diffusion.cpp failed to create a context");
	}
	return makeOk("stable-diffusion.cpp context loaded");
#else
	(void)settings;
	return makeError("stable-diffusion.cpp native backend is not enabled. Run scripts/build-stable-diffusion.*, then compile with OFXGGMLDIFFUSION_WITH_STABLE_DIFFUSION.");
#endif
}

ofxGgmlDiffusionResult ofxGgmlDiffusionNativeBackend::generate(const ofxGgmlDiffusionRequest& request) {
	const auto validation = ofxGgmlDiffusionUtils::validate(request);
	if (!validation) {
		return makeError(joinValidationErrors(validation));
	}
	if (!impl || !impl->isLoaded()) {
		return makeError("stable-diffusion.cpp context is not loaded");
	}
	if (request.mode != ofxGgmlDiffusionMode::TextToImage) {
		return makeError("the first native bridge supports text-to-image requests only");
	}
	if (request.identityAdapter.isConfigured()) {
		return makeError("identity adapters are planned for this addon but are not wired to the native stable-diffusion.cpp bridge yet");
	}

#if OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION
	const auto start = std::chrono::steady_clock::now();
	const std::string prompt = ofxGgmlDiffusionUtils::cleanPrompt(request.prompt);
	std::vector<sd_lora_t> nativeLoras;
	nativeLoras.reserve(request.loras.size());
	for (const auto& lora : request.loras) {
		if (lora.isConfigured()) {
			nativeLoras.push_back({lora.highNoise, lora.strength, lora.path.c_str()});
		}
	}

	sd_img_gen_params_t params{};
	sd_img_gen_params_init(&params);
	params.prompt = emptyToNull(prompt);
	params.negative_prompt = emptyToNull(request.negativePrompt);
	params.width = request.width;
	params.height = request.height;
	params.seed = request.seed;
	params.batch_count = request.batchCount;
	params.loras = nativeLoras.empty() ? nullptr : nativeLoras.data();
	params.lora_count = static_cast<std::uint32_t>(nativeLoras.size());
	params.vae_tiling_params = makeTilingParams(impl->settings.vaeTiling);

	if (request.steps > 0) {
		params.sample_params.sample_steps = request.steps;
	}
	if (!ofxGgmlDiffusionUtils::isAutoValue(request.cfgScale)) {
		params.sample_params.guidance.txt_cfg = request.cfgScale;
	}
	if (!ofxGgmlDiffusionUtils::isAutoValue(request.strength)) {
		params.strength = request.strength;
	}
	if (!ofxGgmlDiffusionUtils::isAutoValue(request.flowShift)) {
		params.sample_params.flow_shift = request.flowShift;
	}
	const auto scheduler = toNativeScheduler(request.scheduler);
	if (scheduler != SCHEDULER_COUNT) {
		params.sample_params.scheduler = scheduler;
	}

	sd_image_t* generated = generate_image(impl->context, &params);
	if (!generated) {
		return makeError("stable-diffusion.cpp returned no images");
	}

	ofxGgmlDiffusionResult result;
	result.success = true;
	result.seed = params.seed;
	result.images.reserve(static_cast<std::size_t>(request.batchCount));
	for (int i = 0; i < request.batchCount; ++i) {
		auto image = copyImage(generated[i]);
		if (image.isAllocated()) {
			result.images.push_back(std::move(image));
		}
	}
	releaseImageArray(generated, request.batchCount);

	const auto elapsed = std::chrono::steady_clock::now() - start;
	result.elapsedMs = std::chrono::duration<float, std::milli>(elapsed).count();
	result.text = "generated " + std::to_string(result.images.size()) + " image(s)";
	if (result.images.empty()) {
		return makeError("stable-diffusion.cpp generated empty image data");
	}
	return result;
#else
	(void)request;
	return makeError("stable-diffusion.cpp native backend is not enabled. Run scripts/build-stable-diffusion.*, then compile with OFXGGMLDIFFUSION_WITH_STABLE_DIFFUSION.");
#endif
}

void ofxGgmlDiffusionNativeBackend::close() {
	if (impl) {
		impl->close();
	}
}

#include "ofxGgmlDiffusionNativeBackend.h"

#include "ofxGgmlDiffusionUtils.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <vector>

#if defined(OFXGGMLDIFFUSION_WITH_STABLE_DIFFUSION) && __has_include(<stable-diffusion.h>)
	#include <stable-diffusion.h>
	#define OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION 1
#else
	#define OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION 0
#endif

#if __has_include("ofPixels.h") && __has_include("ofImage.h")
	#include "ofxGgmlDiffusionImageUtils.h"
	#define OFXGGMLDIFFUSION_HAS_OF_IMAGE_UTILS 1
#else
	#define OFXGGMLDIFFUSION_HAS_OF_IMAGE_UTILS 0
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

	bool appendNativeImage(
		const ofxGgmlDiffusionImage& source,
		std::vector<std::vector<std::uint8_t>>& pixelStorage,
		std::vector<sd_image_t>& nativeImages) {
		if (!source.isAllocated() || (source.channels != 3 && source.channels != 4)) {
			return false;
		}
		const std::size_t byteSize =
			static_cast<std::size_t>(source.width) *
			static_cast<std::size_t>(source.height) *
			static_cast<std::size_t>(source.channels);
		if (source.pixels.size() != byteSize) {
			return false;
		}
		pixelStorage.push_back(source.pixels);
		auto& pixels = pixelStorage.back();
		sd_image_t image{};
		image.width = static_cast<std::uint32_t>(source.width);
		image.height = static_cast<std::uint32_t>(source.height);
		image.channel = static_cast<std::uint32_t>(source.channels);
		image.data = pixels.data();
		nativeImages.push_back(image);
		return true;
	}
#endif
}

bool ofxGgmlDiffusionNativeCapabilities::supportsPhotoMaker() const {
	return stableDiffusionEnabled && photoMakerContextPath && photoMakerImageParams;
}

std::string ofxGgmlDiffusionNativeCapabilities::describe() const {
	std::ostringstream stream;
	stream << "stable-diffusion.cpp "
		   << (stableDiffusionEnabled ? "enabled" : "disabled")
		   << ", PhotoMaker context "
		   << (photoMakerContextPath ? "available" : "missing")
		   << ", PhotoMaker image params "
		   << (photoMakerImageParams ? "available" : "missing");
	return stream.str();
}

ofxGgmlDiffusionNativeCapabilities ofxGgmlDiffusionGetNativeCapabilities() {
	ofxGgmlDiffusionNativeCapabilities capabilities;
	capabilities.stableDiffusionEnabled = OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION != 0;
#if OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION
	sd_ctx_params_t contextParams{};
	sd_ctx_params_init(&contextParams);
	contextParams.photo_maker_path = nullptr;

	sd_img_gen_params_t imageParams{};
	sd_img_gen_params_init(&imageParams);
	imageParams.pm_params.id_images = nullptr;
	imageParams.pm_params.id_images_count = 0;
	imageParams.pm_params.id_embed_path = nullptr;
	imageParams.pm_params.style_strength = 1.0f;

	capabilities.photoMakerContextPath = contextParams.photo_maker_path == nullptr;
	capabilities.photoMakerImageParams =
		imageParams.pm_params.id_images == nullptr &&
		imageParams.pm_params.id_images_count == 0 &&
		imageParams.pm_params.id_embed_path == nullptr &&
		imageParams.pm_params.style_strength == 1.0f;
#endif
	return capabilities;
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

ofxGgmlDiffusionBackendFamily ofxGgmlDiffusionNativeBackend::getBackendFamily() const {
	return ofxGgmlDiffusionBackendFamily::Diffusion;
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
	params.photo_maker_path = emptyToNull(settings.photoMakerPath);
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
	return makeError("stable-diffusion.cpp native backend is not enabled. Run scripts/build-stable-diffusion.* to install the runtime and enable OFXGGMLDIFFUSION_WITH_STABLE_DIFFUSION automatically.");
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
	if (request.mode != ofxGgmlDiffusionMode::TextToImage &&
		request.mode != ofxGgmlDiffusionMode::ImageToVideo) {
		return makeError("the first native bridge supports text-to-image and image-to-video requests only");
	}

#if OFXGGMLDIFFUSION_HAS_STABLE_DIFFUSION
	const auto start = std::chrono::steady_clock::now();
	const std::string prompt = ofxGgmlDiffusionUtils::cleanPrompt(request.prompt);
	const auto scheduler = toNativeScheduler(request.scheduler);
	const auto sampleSteps = request.steps;
	const bool hasSteps = sampleSteps > 0;
	const bool hasCfgScale = !ofxGgmlDiffusionUtils::isAutoValue(request.cfgScale);
	const bool hasStrength = !ofxGgmlDiffusionUtils::isAutoValue(request.strength);
	const bool hasFlowShift = !ofxGgmlDiffusionUtils::isAutoValue(request.flowShift);

	std::vector<sd_lora_t> nativeLoras;
	nativeLoras.reserve(request.loras.size());
	for (const auto& lora : request.loras) {
		if (lora.isConfigured()) {
			nativeLoras.push_back({lora.highNoise, lora.strength, lora.path.c_str()});
		}
	}

	if (request.mode == ofxGgmlDiffusionMode::ImageToVideo) {
#if OFXGGMLDIFFUSION_HAS_OF_IMAGE_UTILS
		ofxGgmlDiffusionImage initImage;
		if (!ofxGgmlDiffusionImageUtils::loadImage(request.initImagePath, initImage)) {
			return makeError("failed to load initImagePath for image-to-video");
		}

		std::vector<std::vector<std::uint8_t>> nativeInitPixels;
		std::vector<sd_image_t> nativeInitImages;
		nativeInitPixels.reserve(1);
		nativeInitImages.reserve(1);
		if (!appendNativeImage(initImage, nativeInitPixels, nativeInitImages)) {
			return makeError("image-to-video initImagePath must be allocated RGB or RGBA image data");
		}

		sd_vid_gen_params_t params{};
		sd_vid_gen_params_init(&params);
		params.prompt = emptyToNull(prompt);
		params.negative_prompt = emptyToNull(request.negativePrompt);
		params.init_image = nativeInitImages.front();
		params.width = request.width;
		params.height = request.height;
		params.seed = request.seed;
		params.video_frames = request.videoFrameCount;
		params.vae_tiling_params = makeTilingParams(impl->settings.vaeTiling);
		params.loras = nativeLoras.empty() ? nullptr : nativeLoras.data();
		params.lora_count = static_cast<std::uint32_t>(nativeLoras.size());
		if (hasSteps) {
			params.sample_params.sample_steps = sampleSteps;
		}
		if (hasCfgScale) {
			params.sample_params.guidance.txt_cfg = request.cfgScale;
		}
		if (hasStrength) {
			params.strength = request.strength;
		}
		if (hasFlowShift) {
			params.sample_params.flow_shift = request.flowShift;
		}
		if (scheduler != SCHEDULER_COUNT) {
			params.sample_params.scheduler = scheduler;
		}

		int numFrames = 0;
		sd_image_t* generated = generate_video(impl->context, &params, &numFrames);
		if (!generated) {
			return makeError("stable-diffusion.cpp returned no frames");
		}

		ofxGgmlDiffusionResult result;
		result.success = true;
		result.seed = params.seed;
		result.videoFrameCount = request.videoFrameCount;
		result.images.reserve(static_cast<std::size_t>(numFrames));
		for (int i = 0; i < numFrames; ++i) {
			auto image = copyImage(generated[i]);
			if (image.isAllocated()) {
				result.images.push_back(std::move(image));
			}
		}
		releaseImageArray(generated, numFrames);
		result.videoFrameCount = static_cast<int>(result.images.size());
		result.elapsedMs = std::chrono::duration<float, std::milli>(
			std::chrono::steady_clock::now() - start).count();
		result.text = "generated " + std::to_string(result.images.size()) + " frame(s)";
		if (result.images.empty()) {
			return makeError("stable-diffusion.cpp generated empty video data");
		}
		return result;
#else
		return makeError("image-to-video requires openFrameworks image utilities in the include path");
#endif
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

	std::vector<std::vector<std::uint8_t>> nativeIdentityPixels;
	std::vector<sd_image_t> nativeIdentityImages;
	if (request.identityAdapter.isConfigured()) {
		const auto capabilities = ofxGgmlDiffusionGetNativeCapabilities();
		if (!capabilities.supportsPhotoMaker()) {
			return makeError("stable-diffusion.cpp PhotoMaker API is not available: " + capabilities.describe());
		}
		if (impl->settings.photoMakerPath.empty()) {
			return makeError("PhotoMaker requires context settings.photoMakerPath before setup");
		}
		if (request.identityAdapter.modelPath != impl->settings.photoMakerPath) {
			return makeError("PhotoMaker request modelPath must match the loaded context photoMakerPath");
		}
		if (request.identityAdapter.referenceImages.empty()) {
			return makeError("PhotoMaker native generation requires decoded referenceImages; load referenceImagePaths with ofxGgmlDiffusionImageUtils::loadImage first");
		}
		nativeIdentityPixels.reserve(request.identityAdapter.referenceImages.size());
		nativeIdentityImages.reserve(request.identityAdapter.referenceImages.size());
		for (const auto& image : request.identityAdapter.referenceImages) {
			if (!appendNativeImage(image, nativeIdentityPixels, nativeIdentityImages)) {
				return makeError("PhotoMaker referenceImages must be allocated RGB or RGBA images");
			}
		}
		params.pm_params.id_images = nativeIdentityImages.data();
		params.pm_params.id_images_count = static_cast<int>(nativeIdentityImages.size());
		params.pm_params.style_strength = request.identityAdapter.strength;
	}
	if (hasSteps) {
		params.sample_params.sample_steps = sampleSteps;
	}
	if (hasCfgScale) {
		params.sample_params.guidance.txt_cfg = request.cfgScale;
	}
	if (hasStrength) {
		params.strength = request.strength;
	}
	if (hasFlowShift) {
		params.sample_params.flow_shift = request.flowShift;
	}
	if (scheduler != SCHEDULER_COUNT) {
		params.sample_params.scheduler = scheduler;
	}

	ofxGgmlDiffusionResult result;
	result.success = true;
	result.seed = params.seed;
	result.images.reserve(static_cast<std::size_t>(request.batchCount));
	sd_image_t* generated = generate_image(impl->context, &params);
	if (!generated) {
		return makeError("stable-diffusion.cpp returned no images");
	}
	for (int i = 0; i < request.batchCount; ++i) {
		auto image = copyImage(generated[i]);
		if (image.isAllocated()) {
			result.images.push_back(std::move(image));
		}
	}
	releaseImageArray(generated, request.batchCount);
	if (!result.images.empty()) {
		const auto elapsed = std::chrono::steady_clock::now() - start;
		result.elapsedMs = std::chrono::duration<float, std::milli>(elapsed).count();
		result.text = "generated " + std::to_string(result.images.size()) + " image(s)";
		return result;
	}
	return makeError("stable-diffusion.cpp generated empty image data");
#else
	(void)request;
	return makeError("stable-diffusion.cpp native backend is not enabled. Run scripts/build-stable-diffusion.* to install the runtime and enable OFXGGMLDIFFUSION_WITH_STABLE_DIFFUSION automatically.");
#endif
}

void ofxGgmlDiffusionNativeBackend::close() {
	if (impl) {
		impl->close();
	}
}

std::unique_ptr<ofxGgmlDiffusionImageGenerationBackend>
ofxGgmlMakeNativeDiffusionImageGenerationBackend() {
	return std::make_unique<ofxGgmlDiffusionNativeBackend>();
}

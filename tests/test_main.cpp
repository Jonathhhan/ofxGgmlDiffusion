#include "ofxGgmlDiffusion/ofxGgmlDiffusionAsyncRunner.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionImageGenerationBackend.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionNativeBackend.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionTinyGanBackend.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionGgufGanBackend.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionTinyGanTraining.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionUtils.h"
#include "ofxGgmlDiffusionVersion.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
	if (OFXGGML_DIFFUSION_VERSION_MAJOR != 1 ||
		OFXGGML_DIFFUSION_VERSION_MINOR != 0 ||
		OFXGGML_DIFFUSION_VERSION_PATCH != 1 ||
		std::string(OFXGGML_DIFFUSION_VERSION_STRING) != "1.0.1" ||
		std::string(ofxGgmlDiffusionGetVersionString()) != "1.0.1") {
		std::cerr << "unexpected diffusion addon version metadata\n";
		return 1;
	}

	ofxGgmlDiffusionRequest request;
	if (ofxGgmlDiffusionUtils::hasInput(request)) {
		std::cerr << "empty request reported as configured\n";
		return 1;
	}

	request.prompt = "a small generative texture study";
	if (!ofxGgmlDiffusionUtils::hasInput(request)) {
		std::cerr << "configured request reported as empty\n";
		return 1;
	}
	request.width = 512;
	request.height = 512;

	const auto description = ofxGgmlDiffusionUtils::describe(request);
	if (description.find(request.prompt) == std::string::npos) {
		std::cerr << "description did not include request input\n";
		return 1;
	}
	if (description.find("512x512") == std::string::npos) {
		std::cerr << "description did not include dimensions\n";
		return 1;
	}

	const auto valid = ofxGgmlDiffusionUtils::validate(request);
	if (!valid.isOk()) {
		std::cerr << "valid request failed validation\n";
		return 1;
	}

	auto invalid = ofxGgmlDiffusionUtils::makeTextToImageRequest("  a   cat  ");
	invalid.width = 513;
	const auto invalidResult = ofxGgmlDiffusionUtils::validate(invalid);
	if (!invalidResult.isError()) {
		std::cerr << "invalid dimension passed validation\n";
		return 1;
	}
	if (invalid.prompt != "a cat") {
		std::cerr << "prompt was not cleaned\n";
		return 1;
	}
	auto videoRequest = ofxGgmlDiffusionUtils::makeImageToVideoRequest(
		"motion shot from skyline",
		"fixtures/video-init.png",
		8);
	videoRequest.width = 512;
	videoRequest.height = 512;
	if (videoRequest.mode != ofxGgmlDiffusionMode::ImageToVideo ||
		videoRequest.videoFrameCount != 8) {
		std::cerr << "image-to-video helper did not configure mode or frame count\n";
		return 1;
	}
	if (!ofxGgmlDiffusionUtils::validate(videoRequest).isOk()) {
		std::cerr << "valid image-to-video request failed validation\n";
		return 1;
	}
	auto invalidVideoRequest = videoRequest;
	invalidVideoRequest.batchCount = 2;
	if (ofxGgmlDiffusionUtils::validate(invalidVideoRequest).isOk()) {
		std::cerr << "image-to-video with batchCount > 1 passed validation\n";
		return 1;
	}
	invalidVideoRequest = videoRequest;
	invalidVideoRequest.videoFrameCount = 0;
	if (ofxGgmlDiffusionUtils::validate(invalidVideoRequest).isOk()) {
		std::cerr << "image-to-video with videoFrameCount 0 passed validation\n";
		return 1;
	}
	invalidVideoRequest = videoRequest;
	invalidVideoRequest.identityAdapter = ofxGgmlDiffusionUtils::makePhotoMakerAdapter(
		"models/photomaker.safetensors",
		{"references/person-01.jpg"},
		"img");
	invalidVideoRequest.identityAdapter.referenceImages = {
		[] {
			ofxGgmlDiffusionImage image;
			image.width = 2;
			image.height = 2;
			image.channels = 3;
			image.pixels.resize(12, 255);
			return image;
		}() };
	if (ofxGgmlDiffusionUtils::validate(invalidVideoRequest).isOk()) {
		std::cerr << "image-to-video with identity adapter passed validation\n";
		return 1;
	}

	auto identityRequest = ofxGgmlDiffusionUtils::makeTextToImageRequest("portrait of img in soft light");
	identityRequest.modelFamily = ofxGgmlDiffusionModelFamily::SDXL;
	identityRequest.identityAdapter = ofxGgmlDiffusionUtils::makePhotoMakerAdapter(
		"models/photomaker.safetensors",
		{"references/person-01.jpg", "references/person-02.jpg"},
		" img ");
	if (!identityRequest.identityAdapter.isConfigured() ||
		identityRequest.identityAdapter.triggerWord != "img" ||
		ofxGgmlDiffusionUtils::getIdentityAdapterTypeName(identityRequest.identityAdapter.type) != "photomaker") {
		std::cerr << "PhotoMaker adapter helper failed\n";
		return 1;
	}
	const auto identityDescription = ofxGgmlDiffusionUtils::describe(identityRequest);
	if (identityDescription.find("photomaker") == std::string::npos) {
		std::cerr << "description did not include identity adapter\n";
		return 1;
	}
	if (ofxGgmlDiffusionUtils::validate(identityRequest).isError()) {
		std::cerr << "valid PhotoMaker request failed validation\n";
		return 1;
	}
	auto invalidIdentity = identityRequest;
	invalidIdentity.identityAdapter.referenceImagePaths.clear();
	if (ofxGgmlDiffusionUtils::validate(invalidIdentity).isOk()) {
		std::cerr << "PhotoMaker request without references passed validation\n";
		return 1;
	}
	auto decodedIdentity = identityRequest;
	decodedIdentity.identityAdapter.referenceImagePaths.clear();
	ofxGgmlDiffusionImage referenceImage;
	referenceImage.width = 2;
	referenceImage.height = 2;
	referenceImage.channels = 3;
	referenceImage.pixels.resize(2 * 2 * 3, 127);
	decodedIdentity.identityAdapter.referenceImages.push_back(referenceImage);
	if (ofxGgmlDiffusionUtils::validate(decodedIdentity).isError()) {
		std::cerr << "PhotoMaker request with decoded reference image failed validation\n";
		return 1;
	}
	auto invalidDecodedIdentity = decodedIdentity;
	invalidDecodedIdentity.identityAdapter.referenceImages.front().channels = 2;
	if (ofxGgmlDiffusionUtils::validate(invalidDecodedIdentity).isOk()) {
		std::cerr << "PhotoMaker request with invalid decoded reference image passed validation\n";
		return 1;
	}
	ofxGgmlDiffusionContextSettings identityContext;
	identityContext.modelPath = "models/sdxl.safetensors";
	identityContext.photoMakerPath = identityRequest.identityAdapter.modelPath;
	if (!identityContext.hasAnyModelPath() ||
		identityContext.photoMakerPath != "models/photomaker.safetensors") {
		std::cerr << "PhotoMaker context settings failed\n";
		return 1;
	}
	const auto nativeCapabilities = ofxGgmlDiffusionGetNativeCapabilities();
	if (nativeCapabilities.stableDiffusionEnabled ||
		nativeCapabilities.supportsPhotoMaker() ||
		nativeCapabilities.describe().find("disabled") == std::string::npos) {
		std::cerr << "default native capabilities were unexpected: "
				  << nativeCapabilities.describe() << "\n";
		return 1;
	}

	auto inpaint = request;
	inpaint.mode = ofxGgmlDiffusionMode::Inpainting;
	if (ofxGgmlDiffusionUtils::validate(inpaint).isOk()) {
		std::cerr << "inpainting without images passed validation\n";
		return 1;
	}

	auto ganRequest = ofxGgmlDiffusionUtils::makeGanImageRequest(
		"small monochrome icon set",
		"models/icon-generator.gguf");
	ganRequest.outputPath = "renders/icons.png";
	ganRequest.seed = 123;
	ganRequest.gan.latentSize = 256;
	ganRequest.gan.truncation = 0.75f;
	if (ofxGgmlDiffusionUtils::getBackendFamilyName(ganRequest.backendFamily) != "gan" ||
		!ganRequest.gan.isConfigured() ||
		ofxGgmlDiffusionUtils::validate(ganRequest).isError()) {
		std::cerr << "valid GAN request failed helpers or validation\n";
		return 1;
	}
	const auto ganDescription = ofxGgmlDiffusionUtils::describe(ganRequest);
	if (ganDescription.find("gan text-to-image") == std::string::npos ||
		ganDescription.find("models/icon-generator.gguf") == std::string::npos) {
		std::cerr << "GAN description did not include backend or generator path\n";
		return 1;
	}
	auto invalidGan = ganRequest;
	invalidGan.gan.generatorPath.clear();
	if (ofxGgmlDiffusionUtils::validate(invalidGan).isOk()) {
		std::cerr << "GAN request without generator path passed validation\n";
		return 1;
	}

	auto preset = ofxGgmlDiffusionMakeDefaultTinyGanPreset();
	preset.latentSize = 64;
	preset.hiddenSize = 32;
	const std::string presetPath = "tiny-gan-test.ofxggmlgan";
	{
		std::ofstream presetFile(presetPath);
		presetFile << ofxGgmlDiffusionSerializeTinyGanPreset(preset);
	}
	ofxGgmlDiffusionTinyGanPreset loadedPreset;
	std::string presetError;
	if (!ofxGgmlDiffusionLoadTinyGanPreset(presetPath, loadedPreset, presetError) ||
		loadedPreset.latentSize != 64 ||
		loadedPreset.hiddenSize != 32 ||
		loadedPreset.architecture != "tiny-mlp") {
		std::cerr << "tiny GAN preset load failed: " << presetError << "\n";
		std::remove(presetPath.c_str());
		return 1;
	}
	auto ganBackend = ofxGgmlMakeUnavailableDiffusionImageGenerationBackend(
		ofxGgmlDiffusionBackendFamily::GAN,
		"gan");
	if (!ganBackend ||
		ganBackend->getBackendName() != "gan" ||
		ganBackend->getBackendFamily() != ofxGgmlDiffusionBackendFamily::GAN ||
		ganBackend->isAvailable() ||
		ganBackend->isLoaded()) {
		std::cerr << "unavailable GAN backend reported unexpected state\n";
		return 1;
	}
	const auto ganSetup = ganBackend->setup(ofxGgmlDiffusionContextSettings{});
	if (ganSetup.isOk() ||
		ganSetup.error.find("not available") == std::string::npos) {
		std::cerr << "unavailable GAN setup was unexpected\n";
		return 1;
	}
	const auto ganResult = ganBackend->generate(ganRequest);
	if (ganResult.isOk() ||
		ganResult.seed != ganRequest.seed ||
		ganResult.outputPath != ganRequest.outputPath ||
		ganResult.references.empty()) {
		std::cerr << "unavailable GAN generation result was unexpected\n";
		return 1;
	}

	ofxGgmlDiffusionTinyGanBackend tinyGan;
	if (tinyGan.getBackendName() != "tiny-ggml-gan" ||
		tinyGan.getBackendFamily() != ofxGgmlDiffusionBackendFamily::GAN ||
		tinyGan.isLoaded()) {
		std::cerr << "tiny GAN backend reported unexpected initial state\n";
		return 1;
	}
	const auto tinySetup = tinyGan.setup(ofxGgmlDiffusionContextSettings{});
	if (tinyGan.isAvailable()) {
		if (!tinySetup || !tinyGan.isLoaded()) {
			std::cerr << "available tiny GAN backend did not set up\n";
			return 1;
		}
		auto tinyRequest = ganRequest;
		tinyRequest.width = 64;
		tinyRequest.height = 64;
		tinyRequest.gan.generatorPath = presetPath;
		const auto tinyResult = tinyGan.generate(tinyRequest);
		if (!tinyResult ||
			tinyResult.images.empty() ||
			!tinyResult.images.front().isAllocated() ||
			tinyResult.images.front().width != 64 ||
			tinyResult.images.front().height != 64) {
			std::cerr << "available tiny GAN backend did not generate pixels\n";
			return 1;
		}
	} else if (tinySetup.isOk() || tinyGan.isLoaded()) {
		std::cerr << "unavailable tiny GAN backend unexpectedly set up\n";
		std::remove(presetPath.c_str());
		return 1;
	}
	std::remove(presetPath.c_str());

	ofxGgmlDiffusionGgufGanBackend ggufGan;
	if (ggufGan.getBackendName() != "ggml-gguf-gan" ||
		ggufGan.getBackendFamily() != ofxGgmlDiffusionBackendFamily::GAN ||
		ggufGan.isLoaded()) {
		std::cerr << "GGUF GAN backend reported unexpected initial state\n";
		return 1;
	}
	const auto missingModelSetup = ggufGan.setup(ofxGgmlDiffusionContextSettings{});
	if (missingModelSetup.isOk() || ggufGan.isLoaded()) {
		std::cerr << "GGUF GAN backend should reject missing model path in setup\n";
		return 1;
	}
	ofxGgmlDiffusionContextSettings wrongExtSettings;
	wrongExtSettings.modelPath = "tiny-mlp.ofxggmlgan";
	const auto wrongExtSetup = ggufGan.setup(wrongExtSettings);
	if (wrongExtSetup.isOk() || ggufGan.isLoaded()) {
		std::cerr << "GGUF GAN backend should reject non-GGUF generator path\n";
		return 1;
	}
	std::ofstream fakeGguf("fake-generator.gguf", std::ios::binary);
	fakeGguf << "not-a-gguf-file";
	fakeGguf.close();
	ofxGgmlDiffusionContextSettings invalidModelSettings;
	invalidModelSettings.modelPath = "fake-generator.gguf";
	const auto invalidModelSetup = ggufGan.setup(invalidModelSettings);
	if (invalidModelSetup.isOk()) {
		std::cerr << "invalid GGUF model should not setup production GAN backend\n";
		std::remove("fake-generator.gguf");
		return 1;
	}
	if (ggufGan.isLoaded()) {
		std::cerr << "GGUF GAN backend remained loaded after invalid model setup\n";
		std::remove("fake-generator.gguf");
		return 1;
	}
	const auto invalidGenResult = ggufGan.generate(ganRequest);
	if (invalidGenResult.isOk()) {
		std::cerr << "GGUF GAN backend generated before successful setup\n";
		std::remove("fake-generator.gguf");
		return 1;
	}
	std::remove("fake-generator.gguf");

	if (const char * pixelGanModel = std::getenv("OFXGGML_PIXEL_GAN_MODEL")) {
		ofxGgmlDiffusionGgufGanBackend pixelGan;
		ofxGgmlDiffusionContextSettings pixelSettings;
		pixelSettings.modelPath = pixelGanModel;
		const auto pixelSetup = pixelGan.setup(pixelSettings);
		if (!pixelSetup || !pixelGan.isLoaded()) {
			std::cerr << "Pixel/DCGAN GGUF backend setup failed: " << pixelSetup.error << "\n";
			return 1;
		}
		auto pixelRequest = ganRequest;
		pixelRequest.width = 64;
		pixelRequest.height = 64;
		pixelRequest.seed = 1234;
		pixelRequest.gan.generatorPath = pixelGanModel;
		pixelRequest.gan.latentSize = 100;
		pixelRequest.gan.truncation = 0.85f;
		const auto pixelResult = pixelGan.generate(pixelRequest);
		if (!pixelResult ||
			pixelResult.images.empty() ||
			!pixelResult.images.front().isAllocated() ||
			pixelResult.images.front().width != 64 ||
			pixelResult.images.front().height != 64 ||
			pixelResult.images.front().channels != 4) {
			std::cerr << "Pixel/DCGAN GGUF backend did not generate RGBA pixels: "
					  << pixelResult.error << "\n";
			return 1;
		}
	}

	const std::filesystem::path datasetPath = "tiny-gan-dataset-test";
	std::filesystem::remove_all(datasetPath);
	std::string fixtureError;
	if (!ofxGgmlDiffusionWriteTinyGanFixtureDataset(datasetPath.string(), 3, fixtureError)) {
		std::cerr << "tiny GAN fixture dataset write failed: " << fixtureError << "\n";
		return 1;
	}
	std::ofstream(datasetPath / "notes.txt") << "metadata";
	const auto datasetScan = ofxGgmlDiffusionScanTinyGanDataset(datasetPath.string());
	if (!datasetScan.exists ||
		!datasetScan.isDirectory ||
		datasetScan.imageCount != 3 ||
		datasetScan.unsupportedFileCount != 1 ||
		datasetScan.imagePaths.size() != 3) {
		std::cerr << "tiny GAN dataset scan failed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	ofxGgmlDiffusionTinyGanImageSample sample;
	std::string sampleError;
	if (!ofxGgmlDiffusionLoadTinyGanPpmImage(datasetScan.imagePaths.front(), sample, sampleError) ||
		!sample.isAllocated() ||
		sample.width != 64 ||
		sample.height != 64 ||
		sample.channels != 3 ||
		sample.pixels.size() != 64 * 64 * 3 ||
		sample.normalizedPixels.size() != sample.pixels.size()) {
		std::cerr << "tiny GAN PPM fixture load failed: " << sampleError << "\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	const auto normalized = ofxGgmlDiffusionNormalizeTinyGanImage(sample);
	if (normalized.size() != sample.normalizedPixels.size() ||
		std::fabs(normalized.front() - sample.normalizedPixels.front()) > 0.0001f ||
		sample.normalizedPixels.front() < -1.0f ||
		sample.normalizedPixels.front() > 1.0f ||
		sample.normalizedPixels[1] < -1.0f ||
		sample.normalizedPixels[1] > 1.0f) {
		std::cerr << "tiny GAN PPM normalization failed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	const auto discriminator = ofxGgmlDiffusionRunTinyGanDiscriminatorForward(sample, 40);
	const auto discriminatorAgain = ofxGgmlDiffusionRunTinyGanDiscriminatorForward(sample, 40);
	if (!discriminator ||
		!discriminatorAgain ||
		discriminator.hiddenSize != 40 ||
		discriminator.probability <= 0.0f ||
		discriminator.probability >= 1.0f ||
		std::fabs(discriminator.probability - discriminatorAgain.probability) > 0.000001f ||
		std::fabs(discriminator.logit - discriminatorAgain.logit) > 0.000001f) {
		std::cerr << "tiny GAN discriminator forward failed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	const auto invalidDiscriminator = ofxGgmlDiffusionRunTinyGanDiscriminatorForward(sample, 4);
	if (invalidDiscriminator.isOk() || invalidDiscriminator.error.empty()) {
		std::cerr << "invalid tiny GAN discriminator forward unexpectedly passed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	const float realGoodLoss = ofxGgmlDiffusionTinyGanBinaryCrossEntropy(0.9f, true);
	const float realBadLoss = ofxGgmlDiffusionTinyGanBinaryCrossEntropy(0.1f, true);
	const float fakeGoodLoss = ofxGgmlDiffusionTinyGanBinaryCrossEntropy(0.1f, false);
	const float fakeBadLoss = ofxGgmlDiffusionTinyGanBinaryCrossEntropy(0.9f, false);
	const float clampedLoss = ofxGgmlDiffusionTinyGanBinaryCrossEntropy(0.0f, true);
	if (!(realGoodLoss < realBadLoss) ||
		!(fakeGoodLoss < fakeBadLoss) ||
		!std::isfinite(clampedLoss) ||
		clampedLoss <= realBadLoss) {
		std::cerr << "tiny GAN BCE helper failed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	const auto lossPair = ofxGgmlDiffusionComputeTinyGanLossPair(0.8f, 0.4f, 0.7f);
	if (lossPair.realLoss <= 0.0f ||
		lossPair.fakeLoss <= 0.0f ||
		lossPair.discriminatorLoss <= 0.0f ||
		lossPair.generatorLoss <= 0.0f ||
		lossPair.generatorLoss >= ofxGgmlDiffusionTinyGanBinaryCrossEntropy(0.3f, true)) {
		std::cerr << "tiny GAN loss pair failed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	const auto updatePreview = ofxGgmlDiffusionPreviewTinyGanWeightUpdate(lossPair, 0.01f);
	if (updatePreview.learningRate != 0.01f ||
		updatePreview.discriminatorGradient == 0.0f ||
		updatePreview.generatorGradient >= 0.0f ||
		updatePreview.updatedDiscriminatorWeight == updatePreview.initialDiscriminatorWeight ||
		updatePreview.updatedGeneratorWeight <= updatePreview.initialGeneratorWeight) {
		std::cerr << "tiny GAN weight update preview failed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	const auto mutatedPreset = ofxGgmlDiffusionApplyTinyGanPresetPreviewUpdate(preset, updatePreview);
	if (mutatedPreset.version != preset.version ||
		mutatedPreset.architecture != preset.architecture ||
		mutatedPreset.latentSize != preset.latentSize ||
		mutatedPreset.hiddenSize != preset.hiddenSize ||
		mutatedPreset.w1Scale == preset.w1Scale ||
		mutatedPreset.w2Scale == preset.w2Scale ||
		mutatedPreset.w1Seed == preset.w1Seed ||
		mutatedPreset.w2Seed == preset.w2Seed) {
		std::cerr << "tiny GAN preset mutation preview failed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	const std::string mutatedPresetPath = "tiny-gan-mutated-test.ofxggmlgan";
	{
		std::ofstream mutatedPresetFile(mutatedPresetPath);
		mutatedPresetFile << ofxGgmlDiffusionSerializeTinyGanPreset(mutatedPreset);
	}
	ofxGgmlDiffusionTinyGanPreset loadedMutatedPreset;
	std::string mutatedPresetError;
	if (!ofxGgmlDiffusionLoadTinyGanPreset(mutatedPresetPath, loadedMutatedPreset, mutatedPresetError) ||
		loadedMutatedPreset.w1Seed != mutatedPreset.w1Seed ||
		loadedMutatedPreset.w2Seed != mutatedPreset.w2Seed) {
		std::cerr << "tiny GAN mutated preset reload failed: " << mutatedPresetError << "\n";
		std::remove(mutatedPresetPath.c_str());
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	std::remove(mutatedPresetPath.c_str());
	const auto missingDatasetScan = ofxGgmlDiffusionScanTinyGanDataset("tiny-gan-missing-dataset-test");
	if (missingDatasetScan.exists || missingDatasetScan.warnings.empty()) {
		std::cerr << "missing tiny GAN dataset scan failed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}

	ofxGgmlDiffusionTinyGanTrainingSettings trainingSettings;
	trainingSettings.datasetPath = datasetPath.string();
	trainingSettings.outputPresetPath = "models/tiny-trained.ofxggmlgan";
	trainingSettings.epochs = 2;
	trainingSettings.batchSize = 8;
	trainingSettings.dryRunBatchesPerEpoch = 3;
	trainingSettings.latentSize = 64;
	trainingSettings.hiddenSize = 32;
	trainingSettings.discriminatorHiddenSize = 40;
	const auto trainingPlan = ofxGgmlDiffusionPlanTinyGanTraining(trainingSettings);
	if (!trainingPlan ||
		trainingPlan.text.find("tiny GAN training dry-run") == std::string::npos ||
		trainingPlan.text.find("tiny-mlp-binary-classifier") == std::string::npos ||
		trainingPlan.outputPresetPath != trainingSettings.outputPresetPath ||
		trainingPlan.epochsPlanned != trainingSettings.epochs ||
		trainingPlan.batchesPerEpoch != trainingSettings.dryRunBatchesPerEpoch ||
		trainingPlan.plannedDiscriminatorUpdates != 6 ||
		trainingPlan.plannedGeneratorUpdates != 6 ||
		trainingPlan.dataset.imageCount != 3 ||
		trainingPlan.dataset.unsupportedFileCount != 1 ||
		trainingPlan.steps.size() != 6 ||
		trainingPlan.finalDiscriminatorLoss <= 0.0f ||
		trainingPlan.finalGeneratorLoss <= 0.0f ||
		trainingPlan.updatePreview.learningRate != trainingSettings.learningRate ||
		trainingPlan.updatePreview.updatedDiscriminatorWeight == trainingPlan.updatePreview.initialDiscriminatorWeight ||
		trainingPlan.updatePreview.updatedGeneratorWeight == trainingPlan.updatePreview.initialGeneratorWeight) {
		std::cerr << "tiny GAN training dry-run plan failed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	const auto dryRunSteps = ofxGgmlDiffusionBuildTinyGanTrainingDryRunSteps(trainingSettings);
	if (dryRunSteps.size() != trainingPlan.steps.size() ||
		dryRunSteps.front().epoch != 1 ||
		dryRunSteps.front().batch != 1 ||
		dryRunSteps.back().epoch != 2 ||
		dryRunSteps.back().batch != 3 ||
		dryRunSteps.front().realProbability <= 0.0f ||
		dryRunSteps.front().fakeProbability <= 0.0f ||
		dryRunSteps.back().discriminatorLoss >= dryRunSteps.front().discriminatorLoss ||
		dryRunSteps.back().generatorLoss >= dryRunSteps.front().generatorLoss) {
		std::cerr << "tiny GAN dry-run step trace failed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	auto invalidTrainingSettings = trainingSettings;
	invalidTrainingSettings.dryRun = false;
	if (ofxGgmlDiffusionRunTinyGanTraining(invalidTrainingSettings).isOk()) {
		std::cerr << "non-dry-run tiny GAN training unexpectedly passed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	invalidTrainingSettings = trainingSettings;
	invalidTrainingSettings.imageWidth = 128;
	if (ofxGgmlDiffusionValidateTinyGanTraining(invalidTrainingSettings).isOk()) {
		std::cerr << "invalid tiny GAN training size unexpectedly passed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	invalidTrainingSettings = trainingSettings;
	invalidTrainingSettings.discriminatorHiddenSize = 4;
	if (ofxGgmlDiffusionValidateTinyGanTraining(invalidTrainingSettings).isOk()) {
		std::cerr << "invalid tiny GAN discriminator size unexpectedly passed\n";
		std::filesystem::remove_all(datasetPath);
		return 1;
	}
	std::filesystem::remove_all(datasetPath);

	ofxGgmlDiffusionImage image;
	image.width = 2;
	image.height = 2;
	image.channels = 3;
	image.pixels.resize(12);
	if (!image.isAllocated() || image.getByteSize() != 12) {
		std::cerr << "image allocation helpers failed\n";
		return 1;
	}

	ofxGgmlDiffusionNativeBackend backend;
	if (backend.getBackendName() != "stable-diffusion.cpp") {
		std::cerr << "unexpected native backend name\n";
		return 1;
	}
	if (backend.getBackendFamily() != ofxGgmlDiffusionBackendFamily::Diffusion) {
		std::cerr << "native backend did not report diffusion family\n";
		return 1;
	}
	auto nativeInterface = ofxGgmlMakeNativeDiffusionImageGenerationBackend();
	if (!nativeInterface ||
		nativeInterface->getBackendName() != "stable-diffusion.cpp" ||
		nativeInterface->getBackendFamily() != ofxGgmlDiffusionBackendFamily::Diffusion) {
		std::cerr << "native backend factory did not return image generation interface\n";
		return 1;
	}
	const auto nativeInterfaceSetup = nativeInterface->setup(ofxGgmlDiffusionContextSettings{});
	if (nativeInterfaceSetup.isOk()) {
		std::cerr << "native interface setup succeeded without model/runtime\n";
		return 1;
	}
	if (backend.isLoaded()) {
		std::cerr << "native backend reported loaded before setup\n";
		return 1;
	}
	const auto setupResult = backend.setup(ofxGgmlDiffusionContextSettings{});
	if (setupResult.isOk()) {
		std::cerr << "native backend setup succeeded without model/runtime\n";
		return 1;
	}
	const auto generateResult = backend.generate(request);
	if (generateResult.isOk()) {
		std::cerr << "native backend generated without setup\n";
		return 1;
	}

	ofxGgmlDiffusionAsyncRunner runner;
	if (runner.getState() != ofxGgmlDiffusionTaskState::Idle ||
		ofxGgmlDiffusionGetTaskStateName(runner.getState()) != "idle") {
		std::cerr << "async runner did not start idle\n";
		return 1;
	}
	const auto startResult = runner.start(ofxGgmlDiffusionContextSettings{}, request);
	if (!startResult.isOk()) {
		std::cerr << "async runner did not start\n";
		return 1;
	}
	runner.wait();
	if (!runner.isDone() || runner.getState() != ofxGgmlDiffusionTaskState::Failed) {
		std::cerr << "async runner did not report failed setup\n";
		return 1;
	}
	ofxGgmlDiffusionResult asyncResult;
	if (!runner.consumeResult(asyncResult)) {
		std::cerr << "async runner did not expose result\n";
		return 1;
	}
	if (asyncResult.isOk() || runner.hasResult()) {
		std::cerr << "async runner result consumption failed\n";
		return 1;
	}

	return 0;
}

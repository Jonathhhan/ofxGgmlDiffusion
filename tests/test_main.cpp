#include "ofxGgmlDiffusion/ofxGgmlDiffusionAsyncRunner.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionImageGenerationBackend.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionNativeBackend.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionTinyGanBackend.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionTinyGanTraining.h"
#include "ofxGgmlDiffusion/ofxGgmlDiffusionUtils.h"

#include <cstdio>
#include <fstream>
#include <iostream>

int main() {
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

	ofxGgmlDiffusionTinyGanTrainingSettings trainingSettings;
	trainingSettings.datasetPath = "datasets/icons";
	trainingSettings.outputPresetPath = "models/tiny-trained.ofxggmlgan";
	trainingSettings.epochs = 2;
	trainingSettings.batchSize = 8;
	trainingSettings.latentSize = 64;
	trainingSettings.hiddenSize = 32;
	const auto trainingPlan = ofxGgmlDiffusionPlanTinyGanTraining(trainingSettings);
	if (!trainingPlan ||
		trainingPlan.text.find("tiny GAN training dry-run") == std::string::npos ||
		trainingPlan.outputPresetPath != trainingSettings.outputPresetPath ||
		trainingPlan.epochsPlanned != trainingSettings.epochs) {
		std::cerr << "tiny GAN training dry-run plan failed\n";
		return 1;
	}
	auto invalidTrainingSettings = trainingSettings;
	invalidTrainingSettings.dryRun = false;
	if (ofxGgmlDiffusionRunTinyGanTraining(invalidTrainingSettings).isOk()) {
		std::cerr << "non-dry-run tiny GAN training unexpectedly passed\n";
		return 1;
	}
	invalidTrainingSettings = trainingSettings;
	invalidTrainingSettings.imageWidth = 128;
	if (ofxGgmlDiffusionValidateTinyGanTraining(invalidTrainingSettings).isOk()) {
		std::cerr << "invalid tiny GAN training size unexpectedly passed\n";
		return 1;
	}

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

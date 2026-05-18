#include "ofApp.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>

namespace {
	const char* backendModeNames[] = {
		"Proof (deterministic tiny)",
		"Production (GGUF model)"
	};

	const char* generatorModeNames[] = {
		"Built-in",
		"Preset",
		"Preview preset"
	};

	std::string getEnvironmentVariable(const std::string& name) {
#if defined(_MSC_VER)
		char* value = nullptr;
		size_t valueSize = 0;
		if (_dupenv_s(&value, &valueSize, name.c_str()) != 0 || value == nullptr) {
			return "";
		}
		std::string result(value);
		std::free(value);
		return result;
#else
		const char* value = std::getenv(name.c_str());
		return value != nullptr ? value : "";
#endif
	}

	std::string firstLine(const std::string& text) {
		const auto lineBreak = text.find('\n');
		if (lineBreak == std::string::npos) {
			return text;
		}
		return text.substr(0, lineBreak);
	}

	bool hasExtensionIgnoreCase(std::string_view value, std::string_view extension) {
		if (value.size() < extension.size()) {
			return false;
		}
		const auto suffix = value.substr(value.size() - extension.size());
		if (suffix.size() != extension.size()) {
			return false;
		}
		return std::equal(
			extension.begin(),
			extension.end(),
			suffix.begin(),
			suffix.end(),
			[](char lhs, char rhs) { return std::tolower(static_cast<unsigned char>(lhs)) == std::tolower(static_cast<unsigned char>(rhs)); });
	}

	std::string toLowerCopy(std::string value) {
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
			return static_cast<char>(std::tolower(ch));
		});
		return value;
	}

	std::string describePathOrNone(const std::string& path) {
		return path.empty() ? "none" : ofFilePath::getFileName(path);
	}

	std::string findFirstGgufGenerator(const std::string& modelsPath) {
		if (!ofFile::doesFileExist(modelsPath, false)) {
			return "";
		}
		ofDirectory models(modelsPath);
		if (!models.isDirectory()) {
			return "";
		}
		models.listDir();
		const std::array<std::string, 2> preferredNames = {
			"pixel-model-f16.gguf",
			"generator.gguf"
		};
		for (const auto& preferredName : preferredNames) {
			for (const auto& file : models.getFiles()) {
				const auto filename = file.getFileName();
				if (toLowerCopy(filename) == preferredName) {
					return file.getAbsolutePath();
				}
			}
		}
		for (const auto& file : models.getFiles()) {
			const auto filename = file.getFileName();
			const auto lowerFilename = toLowerCopy(filename);
			if (hasExtensionIgnoreCase(filename, ".gguf") &&
				(lowerFilename.find("gan") != std::string::npos ||
				 lowerFilename.find("generator") != std::string::npos ||
				 lowerFilename.find("pixel") != std::string::npos)) {
				return file.getAbsolutePath();
			}
		}
		return "";
	}
}

void ofApp::setup() {
	ofSetWindowTitle("ofxGgmlDiffusion GAN example");
	gui.setup();

	prompt = "small monochrome icon set for an openFrameworks tool";
	std::snprintf(promptBuffer.data(), promptBuffer.size(), "%s", prompt.c_str());
	randomizeSeed();

	backendMode = GeneratorBackendMode::Proof;
	productionGeneratorPath = findProductionGeneratorPath();
	if (!productionGeneratorPath.empty() &&
		hasExtensionIgnoreCase(productionGeneratorPath, ".gguf")) {
		backendMode = GeneratorBackendMode::Production;
	}
	backendModeIndex = static_cast<int>(backendMode);
	const auto envGenerator = getEnvironmentVariable("OFXGGML_GAN_GENERATOR");
	if (backendMode == GeneratorBackendMode::Proof &&
		hasExtensionIgnoreCase(envGenerator, ".ofxggmlgan") &&
		ofFile::doesFileExist(envGenerator, false)) {
		generatorMode = GeneratorMode::Preset;
	}
	generatorModeIndex = static_cast<int>(generatorMode);
	rebuildBackend();
	refreshProductionPathBuffer();
	rebuildRequest();
	updateTrainingPlan();
	refreshFixtureTextures();

	status = ofxGgmlDiffusionUtils::describe(request);
	if (backend->isAvailable()) {
		detail = "Proof lane: tiny synthetic ggml-MLP (not trained).";
	} else {
		detail = "Proof lane unavailable. Run ofxGgmlCore scripts\\setup-ggml.bat, then rebuild this example.";
	}
	if (backendMode == GeneratorBackendMode::Production) {
		detail = "Production path: " + describePathOrNone(findProductionGeneratorPath());
	}
	ofLogNotice("ofxGgmlDiffusionGanExample") << status;
	ofLogNotice("ofxGgmlDiffusionGanExample") << detail;
}

void ofApp::rebuildBackend() {
	if (backendMode == GeneratorBackendMode::Production) {
		backend = std::make_unique<ofxGgmlDiffusionGgufGanBackend>();
		settings.modelPath = findProductionGeneratorPath();
		return;
	}
	backend = std::make_unique<ofxGgmlDiffusionTinyGanBackend>();
	settings.modelPath.clear();
}

void ofApp::update() {
	advanceAnimation();
}

void ofApp::keyPressed(int key) {
	if (key == 'r' || key == 'R') {
		runGeneration();
	}
}

void ofApp::runGeneration() {
	if (!lockSeed) {
		randomizeSeed();
	}
	if (backendMode == GeneratorBackendMode::Production) {
		auto useProofFallback = [&]() {
			std::string fallbackPath;
			if (ofFile::doesFileExist(getPreviewPresetPath(), false)) {
				generatorMode = GeneratorMode::PreviewPreset;
				fallbackPath = getPreviewPresetPath();
			} else if (ofFile::doesFileExist(getPresetPath(), false)) {
				generatorMode = GeneratorMode::Preset;
				fallbackPath = getPresetPath();
			}
			if (fallbackPath.empty()) {
				return false;
			}

			backendMode = GeneratorBackendMode::Proof;
			backendModeIndex = static_cast<int>(backendMode);
			generatorModeIndex = static_cast<int>(generatorMode);
			rebuildBackend();
			rebuildRequest();
			status = "using proof preset";
			detail = "No GGUF generator model was found; using proof preset " +
				ofFilePath::getFileName(fallbackPath) + ".";
			ofLogNotice("ofxGgmlDiffusionGanExample") << detail;
			return true;
		};

		productionGeneratorPath = findProductionGeneratorPath();
		settings.modelPath = productionGeneratorPath;
		request.gan.generatorPath = productionGeneratorPath;
		if (settings.modelPath.empty()) {
			if (useProofFallback()) {
				return runGeneration();
			}
			status = "backend unavailable";
			detail = "Production lane requires a supported GGUF generator model. Place one in bin/data/models, use Browse, set OFXGGML_GAN_GENERATOR, run scripts\\download-pixel-gan-model.bat, or switch to Proof.";
			ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
			return;
		}
		if (!ofFile::doesFileExist(settings.modelPath, false)) {
			status = "backend unavailable";
			detail = "Production generator file not found: " + settings.modelPath;
			ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
			return;
		}
		if (!hasExtensionIgnoreCase(settings.modelPath, ".gguf")) {
			if (hasExtensionIgnoreCase(settings.modelPath, ".ofxggmlgan") && useProofFallback()) {
				return runGeneration();
			}
			status = "backend unavailable";
			detail = "Production lane requires a .gguf generator file. Use .ofxggmlgan presets in Proof.";
			ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
			return;
		}
	}
	rebuildRequest();
	const auto validation = ofxGgmlDiffusionUtils::validate(request);
	if (!validation) {
		status = "request invalid";
		detail = validation.errors.empty() ? "GAN request failed validation" : validation.errors.front();
		ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
		return;
	}

	status = "starting";
	ofLogNotice("ofxGgmlDiffusionGanExample") << ofxGgmlDiffusionUtils::describe(request);
	auto setupResult = backend->setup(settings);
	if (!setupResult) {
		status = "backend unavailable";
		detail = setupResult.error;
		ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
		return;
	}

	applyResult(backend->generate(request));
}

void ofApp::drawGui() {
	const float panelWidth = std::min(520.0f, ofGetWidth() * 0.48f);
	ImGui::SetNextWindowPos(ImVec2(16, 16), ImGuiCond_Once);
	ImGui::SetNextWindowSize(ImVec2(panelWidth, ofGetHeight() - 32.0f), ImGuiCond_Once);
	ImGui::Begin("ofxGgmlDiffusion GAN");

	if (ImGui::InputTextMultiline("Prompt", promptBuffer.data(), promptBuffer.size(), ImVec2(-1.0f, 76.0f))) {
		prompt = promptBuffer.data();
		rebuildRequest();
	}

	drawSeedControls();

	if (ImGui::Combo("Backend lane", &backendModeIndex, backendModeNames, 2)) {
		backendMode = static_cast<GeneratorBackendMode>(backendModeIndex);
		rebuildBackend();
		productionGeneratorPath = findProductionGeneratorPath();
		refreshProductionPathBuffer();
		if (backendMode == GeneratorBackendMode::Proof) {
			detail = "Proof lane selected.";
			updateTrainingPlan();
			refreshFixtureTextures();
		} else {
			detail = "Production lane selected. GGUF: " + describePathOrNone(findProductionGeneratorPath());
		}
		rebuildRequest();
	}

	if (backendMode == GeneratorBackendMode::Proof) {
		drawProofControls();
	} else {
		drawProductionControls();
	}

	if (ImGui::Button("Run")) {
		runGeneration();
	}

	ImGui::Separator();
	drawStatus();

	ImGui::Text("Status: %s", status.c_str());
	ImGui::TextWrapped("%s", detail.c_str());
	ImGui::TextWrapped("Generator path: %s", request.gan.generatorPath.c_str());

	ImGui::End();
}

void ofApp::drawSeedControls() {
	ImGui::Separator();
	if (ImGui::Checkbox("Lock seed", &lockSeed)) {
		rebuildRequest();
	}
	ImGui::SameLine();
	if (ImGui::Button("New seed")) {
		randomizeSeed();
		rebuildRequest();
	}
	if (ImGui::InputInt("Seed", &seed)) {
		seed = std::max(1, seed);
		lockSeed = true;
		rebuildRequest();
	}
	ImGui::TextWrapped("%s", lockSeed ? "Seed is locked." : "Run picks a new seed.");

	ImGui::Separator();
	if (ImGui::Checkbox("Animate", &animate)) {
		lockSeed = true;
		lastAnimationTime = 0.0;
		rebuildRequest();
	}
	ImGui::SameLine();
	ImGui::TextWrapped("Seed walk");
	ImGui::SliderFloat("Animation FPS", &animationFps, 0.5f, 12.0f, "%.1f");
	if (ImGui::InputInt("Seed step", &animationSeedStep)) {
		if (animationSeedStep == 0) {
			animationSeedStep = 1;
		}
	}
}

void ofApp::drawProofControls() {
	ImGui::Separator();
	if (ImGui::Combo("Generator", &generatorModeIndex, generatorModeNames, 3)) {
		generatorMode = static_cast<GeneratorMode>(generatorModeIndex);
		rebuildRequest();
	}

	ImGui::Separator();
	bool settingsChanged = false;
	settingsChanged |= ImGui::SliderInt("Fixtures", &fixtureCount, 1, 32);
	settingsChanged |= ImGui::SliderInt("Epochs", &epochs, 1, 16);
	settingsChanged |= ImGui::SliderInt("Batch size", &batchSize, 1, 32);
	settingsChanged |= ImGui::SliderInt("Dry-run batches", &dryRunBatchesPerEpoch, 1, 16);
	settingsChanged |= ImGui::SliderFloat("Learning rate", &learningRate, 0.0001f, 0.01f, "%.4f");
	if (settingsChanged) {
		updateTrainingPlan();
		refreshFixtureTextures();
	}

	if (ImGui::Button("Create fixtures")) {
		createFixtureDataset();
	}
	ImGui::SameLine();
	if (ImGui::Button("Write preview preset")) {
		writePreviewPreset();
	}

	ImGui::Separator();
	if (trainingPlan) {
		ImGui::Text("Dataset images: %d", trainingPlan.dataset.imageCount);
		ImGui::Text("Planned updates: D %d / G %d",
			trainingPlan.plannedDiscriminatorUpdates,
			trainingPlan.plannedGeneratorUpdates);
		ImGui::Text("Preview loss: D %.4f / G %.4f",
			trainingPlan.finalDiscriminatorLoss,
			trainingPlan.finalGeneratorLoss);
	} else {
		ImGui::TextWrapped("Training plan: %s", trainingSummary.c_str());
	}
	if (!trainingSummary.empty()) {
		ImGui::TextWrapped("%s", trainingSummary.c_str());
	}
}

void ofApp::drawProductionControls() {
	ImGui::Separator();
	if (ImGui::InputText("GGUF generator", productionGeneratorPathBuffer.data(), productionGeneratorPathBuffer.size())) {
		productionGeneratorPath = productionGeneratorPathBuffer.data();
		rebuildRequest();
	}
	ImGui::SameLine();
	if (ImGui::Button("Browse")) {
		auto result = ofSystemLoadDialog("Load GAN GGUF", false);
		if (result.bSuccess) {
			productionGeneratorPath = result.getPath();
			refreshProductionPathBuffer();
			rebuildRequest();
		}
	}
	ImGui::SameLine();
	if (ImGui::Button("From environment")) {
		const auto envGenerator = getEnvironmentVariable("OFXGGML_GAN_GENERATOR");
		if (!envGenerator.empty()) {
			productionGeneratorPath = envGenerator;
			refreshProductionPathBuffer();
			rebuildRequest();
		}
	}
	if (productionGeneratorPath.empty()) {
		ImGui::TextWrapped("No generator selected.");
	} else if (!ofFile::doesFileExist(productionGeneratorPath, false)) {
		ImGui::TextWrapped("Generator not found: %s", productionGeneratorPath.c_str());
	} else {
		ImGui::TextWrapped("Generator: %s", productionGeneratorPath.c_str());
	}
	ImGui::TextWrapped(
		"Use a real exported GGUF GAN generator checkpoint for production inference. "
		"This build currently supports the gguf-org/pixel Pixel/DCGAN checkpoint. "
		"Pixel/DCGAN is unconditional, so prompt text is folded into deterministic sample variation rather than semantic text guidance. "
		"The lane is intentionally separated from proof mode so it does not affect training tooling.");
}

void ofApp::drawStatus() {
	if (backendMode == GeneratorBackendMode::Proof) {
		ImGui::TextWrapped("Lane: proof (deterministic, tiny, no quality guarantees)");
		if (!backend->isAvailable()) {
			ImGui::TextWrapped("Run ofxGgmlCore scripts\\setup-ggml.bat then rebuild this example");
		}
	} else {
		ImGui::TextWrapped("Lane: production (GGUF checkpoint)");
		const auto effectiveProductionPath = findProductionGeneratorPath();
		if (effectiveProductionPath.empty()) {
			ImGui::TextWrapped("Waiting for a supported GGUF model. Put pixel-model-f16.gguf in bin/data/models or use Browse / OFXGGML_GAN_GENERATOR.");
		} else if (!ofFile::doesFileExist(effectiveProductionPath, false)) {
			ImGui::TextWrapped("Configured path does not exist: %s", effectiveProductionPath.c_str());
		} else if (!hasExtensionIgnoreCase(effectiveProductionPath, ".gguf")) {
			ImGui::TextWrapped("Configured model is not .gguf: %s", effectiveProductionPath.c_str());
		} else {
			ImGui::TextWrapped("Production model: %s", effectiveProductionPath.c_str());
		}
		if (backend->isAvailable()) {
			ImGui::TextWrapped("Backend runtime is available.");
		}
	}
}

void ofApp::createFixtureDataset() {
	std::string error;
	if (!ofxGgmlDiffusionWriteTinyGanFixtureDataset(getPreviewDatasetPath(), fixtureCount, error)) {
		status = "fixture write failed";
		detail = error;
		ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
		return;
	}

	status = "fixtures ready";
	detail = "Wrote " + ofToString(fixtureCount) + " fixture images to " + getPreviewDatasetPath();
	ofLogNotice("ofxGgmlDiffusionGanExample") << detail;
	updateTrainingPlan();
	refreshFixtureTextures();
}

void ofApp::writePreviewPreset() {
	updateTrainingPlan();
	if (!trainingPlan) {
		status = "preview preset failed";
		detail = trainingPlan.error;
		ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
		return;
	}

	ofxGgmlDiffusionTinyGanPreset preset = ofxGgmlDiffusionMakeDefaultTinyGanPreset();
	if (ofFile::doesFileExist(getPresetPath(), false)) {
		std::string presetError;
		if (!ofxGgmlDiffusionLoadTinyGanPreset(getPresetPath(), preset, presetError)) {
			status = "preset load failed";
			detail = presetError;
			ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
			return;
		}
	}

	const auto previewPreset = ofxGgmlDiffusionApplyTinyGanPresetPreviewUpdate(
		preset,
		trainingPlan.updatePreview);

	const auto outputPath = getPreviewPresetPath();
	ofDirectory::createDirectory(ofFilePath::getEnclosingDirectory(outputPath, false), false, true);
	std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
	if (!output) {
		status = "preset write failed";
		detail = "Could not open " + outputPath;
		ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
		return;
	}
	output << ofxGgmlDiffusionSerializeTinyGanPreset(previewPreset);
	output.close();

	generatorMode = GeneratorMode::PreviewPreset;
	generatorModeIndex = static_cast<int>(generatorMode);
	rebuildRequest();
	status = "preview preset ready";
	detail = "Wrote " + outputPath;
	ofLogNotice("ofxGgmlDiffusionGanExample") << detail;
}

void ofApp::updateTrainingPlan() {
	ofxGgmlDiffusionTinyGanTrainingSettings trainingSettings;
	trainingSettings.datasetPath = getPreviewDatasetPath();
	trainingSettings.outputPresetPath = getPreviewPresetPath();
	trainingSettings.epochs = std::max(1, epochs);
	trainingSettings.batchSize = std::max(1, batchSize);
	trainingSettings.dryRunBatchesPerEpoch = std::max(1, dryRunBatchesPerEpoch);
	trainingSettings.learningRate = std::max(0.0001f, learningRate);
	trainingSettings.dryRun = true;
	trainingPlan = ofxGgmlDiffusionPlanTinyGanTraining(trainingSettings);
	trainingSummary = trainingPlan ? firstLine(trainingPlan.text) : trainingPlan.error;
}

void ofApp::refreshFixtureTextures() {
	fixtureTextures.clear();
	if (!trainingPlan || trainingPlan.dataset.imagePaths.empty()) {
		return;
	}

	const auto sampleCount = std::min<std::size_t>(trainingPlan.dataset.imagePaths.size(), 8);
	for (std::size_t i = 0; i < sampleCount; ++i) {
		ofxGgmlDiffusionTinyGanImageSample sample;
		std::string error;
		if (!ofxGgmlDiffusionLoadTinyGanPpmImage(trainingPlan.dataset.imagePaths[i], sample, error)) {
			ofLogWarning("ofxGgmlDiffusionGanExample") << error;
			continue;
		}

		ofPixels pixels;
		pixels.setFromPixels(sample.pixels.data(), sample.width, sample.height, OF_IMAGE_COLOR);
		if (!pixels.isAllocated()) {
			continue;
		}
		fixtureTextures.emplace_back();
		fixtureTextures.back().loadData(pixels);
	}
}

void ofApp::rebuildRequest() {
	request = ofxGgmlDiffusionUtils::makeGanImageRequest(
		prompt,
		backendMode == GeneratorBackendMode::Production ? findProductionGeneratorPath() : findGeneratorPath());
	request.width = 512;
	request.height = 512;
	request.outputPath = getOutputPath();
	request.seed = seed;
	request.gan.latentSize = backendMode == GeneratorBackendMode::Production ? 100 : 512;
	request.gan.truncation = 0.85f;
}

void ofApp::randomizeSeed() {
	seed = static_cast<int>(ofRandom(1.0f, 2147483000.0f));
	if (seed <= 0) {
		seed = 1;
	}
}

void ofApp::advanceAnimation() {
	if (!animate) {
		return;
	}
	const double now = ofGetElapsedTimef();
	const double interval = 1.0 / std::max(0.5f, animationFps);
	if (lastAnimationTime > 0.0 && now - lastAnimationTime < interval) {
		return;
	}
	lastAnimationTime = now;
	lockSeed = true;
	const long long nextSeed = static_cast<long long>(seed) + static_cast<long long>(animationSeedStep);
	if (nextSeed <= 0) {
		seed = 1;
	} else if (nextSeed > 2147483000LL) {
		seed = 1 + static_cast<int>(nextSeed % 2147483000LL);
	} else {
		seed = static_cast<int>(nextSeed);
	}
	rebuildRequest();
	runGeneration();
}

void ofApp::applyResult(ofxGgmlDiffusionResult result) {
	if (!result) {
		status = "generation failed";
		detail = result.error;
		ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
		return;
	}

	if (!ofxGgmlDiffusionImageUtils::saveFirstImage(result, request.outputPath)) {
		status = "save failed";
		detail = result.error;
		ofLogWarning("ofxGgmlDiffusionGanExample") << detail;
		return;
	}

	ofPixels pixels;
	if (!result.images.empty() &&
		ofxGgmlDiffusionImageUtils::toPixels(result.images.front(), pixels)) {
		texture.loadData(pixels);
	}

	status = "complete";
	detail = "Saved " + result.outputPath + " (seed " + ofToString(result.seed) + ")";
	ofLogNotice("ofxGgmlDiffusionGanExample") << detail;
}

std::string ofApp::findGeneratorPath() const {
	if (backendMode == GeneratorBackendMode::Production) {
		return findProductionGeneratorPath();
	}
	if (generatorMode == GeneratorMode::Preset) {
		return getPresetPath();
	}
	if (generatorMode == GeneratorMode::PreviewPreset) {
		return getPreviewPresetPath();
	}
	return "builtin:tiny-mlp";
}

std::string ofApp::findProductionGeneratorPath() const {
	if (!productionGeneratorPath.empty()) {
		return productionGeneratorPath;
	}
	const auto envGenerator = getEnvironmentVariable("OFXGGML_GAN_GENERATOR");
	if (!envGenerator.empty() &&
		hasExtensionIgnoreCase(envGenerator, ".gguf") &&
		ofFile::doesFileExist(envGenerator, false)) {
		return envGenerator;
	}
	if (!envGenerator.empty() && hasExtensionIgnoreCase(envGenerator, ".gguf")) {
		return envGenerator;
	}
	const auto dataModelsPath = ofToDataPath("models", true);
	const auto discoveredModel = findFirstGgufGenerator(dataModelsPath);
	if (!discoveredModel.empty()) {
		return discoveredModel;
	}
	return "";
}

void ofApp::refreshProductionPathBuffer() {
	std::fill_n(productionGeneratorPathBuffer.data(), productionGeneratorPathBuffer.size(), '\0');
	std::snprintf(productionGeneratorPathBuffer.data(), productionGeneratorPathBuffer.size(), "%s", productionGeneratorPath.c_str());
}

std::string ofApp::getPresetPath() const {
	const auto envGenerator = getEnvironmentVariable("OFXGGML_GAN_GENERATOR");
	if (!envGenerator.empty() && hasExtensionIgnoreCase(envGenerator, ".ofxggmlgan") && ofFile::doesFileExist(envGenerator, false)) {
		return envGenerator;
	}
	return ofToDataPath("models/tiny-mlp.ofxggmlgan", true);
}

std::string ofApp::getPreviewPresetPath() const {
	return ofToDataPath("models/tiny-preview-trained.ofxggmlgan", true);
}

std::string ofApp::getPreviewDatasetPath() const {
	return ofToDataPath("datasets/tiny-fixtures", true);
}

std::string ofApp::getOutputPath() const {
	const auto outputDir = ofToDataPath("outputs", true);
	ofDirectory::createDirectory(outputDir, false, true);
	return ofFilePath::join(outputDir, "ofxGgmlDiffusionGanExample.png");
}

void ofApp::draw() {
	ofBackground(18);

	const float leftMargin = std::min(552.0f, ofGetWidth() * 0.52f);
	const float previewX = leftMargin + 24.0f;
	const float previewY = 32.0f;
	const float previewMaxWidth = std::max(120.0f, ofGetWidth() - previewX - 32.0f);
	const float previewMaxHeight = std::max(120.0f, ofGetHeight() - previewY - 32.0f);
	const float outputSize = std::min({previewMaxWidth, previewMaxHeight * 0.52f, 512.0f});

	ofSetColor(240);
	ofDrawBitmapString("GAN preview", previewX, previewY);
	if (texture.isAllocated()) {
		const float scale = std::min(outputSize / texture.getWidth(), outputSize / texture.getHeight());
		texture.draw(previewX, previewY + 24.0f, texture.getWidth() * scale, texture.getHeight() * scale);
	} else {
		ofSetColor(120);
		ofNoFill();
		ofDrawRectangle(previewX, previewY + 24.0f, outputSize, outputSize);
		ofFill();
		ofSetColor(190);
		ofDrawBitmapString("Press Run or R", previewX + 18.0f, previewY + 54.0f);
	}

	drawTrainingPreview(
		previewX,
		previewY + outputSize + 64.0f,
		previewMaxWidth,
		std::max(120.0f, previewMaxHeight - outputSize - 64.0f));

	gui.begin();
	drawGui();
	gui.end();
}

void ofApp::drawTrainingPreview(float x, float y, float width, float height) {
	ofSetColor(240);
	ofDrawBitmapString("Fixture samples", x, y);

	const float rowY = y + 18.0f;
	const float gap = 8.0f;
	const int maxVisible = fixtureTextures.empty() ? 1 : static_cast<int>(fixtureTextures.size());
	const float thumbnailSize = std::min(64.0f, std::max(36.0f, (width - gap * (maxVisible - 1)) / maxVisible));
	if (fixtureTextures.empty()) {
		ofSetColor(120);
		ofNoFill();
		ofDrawRectangle(x, rowY, thumbnailSize, thumbnailSize);
		ofFill();
		ofSetColor(190);
		ofDrawBitmapString("Create fixtures", x + thumbnailSize + 12.0f, rowY + 22.0f);
	} else {
		for (std::size_t i = 0; i < fixtureTextures.size(); ++i) {
			const float thumbX = x + static_cast<float>(i) * (thumbnailSize + gap);
			fixtureTextures[i].draw(thumbX, rowY, thumbnailSize, thumbnailSize);
		}
	}

	const float curveY = rowY + thumbnailSize + 36.0f;
	const float curveHeight = std::max(72.0f, height - (curveY - y));
	drawLossCurve(x, curveY, width, curveHeight);
}

void ofApp::drawLossCurve(float x, float y, float width, float height) {
	ofSetColor(240);
	ofDrawBitmapString("Dry-run loss", x, y);
	const float plotX = x;
	const float plotY = y + 16.0f;
	const float plotWidth = width;
	const float plotHeight = std::max(48.0f, height - 22.0f);

	ofSetColor(70);
	ofNoFill();
	ofDrawRectangle(plotX, plotY, plotWidth, plotHeight);
	ofFill();

	if (!trainingPlan || trainingPlan.steps.empty()) {
		ofSetColor(190);
		ofDrawBitmapString("No dry-run steps yet", plotX + 12.0f, plotY + 28.0f);
		return;
	}

	float minLoss = std::numeric_limits<float>::max();
	float maxLoss = std::numeric_limits<float>::lowest();
	for (const auto& step : trainingPlan.steps) {
		minLoss = std::min({minLoss, step.discriminatorLoss, step.generatorLoss});
		maxLoss = std::max({maxLoss, step.discriminatorLoss, step.generatorLoss});
	}
	if (maxLoss - minLoss < 0.0001f) {
		maxLoss = minLoss + 1.0f;
	}

	auto mapPoint = [&](std::size_t index, float value) {
		const float t = trainingPlan.steps.size() == 1
			? 0.0f
			: static_cast<float>(index) / static_cast<float>(trainingPlan.steps.size() - 1);
		const float normalized = (value - minLoss) / (maxLoss - minLoss);
		return glm::vec2(
			plotX + t * plotWidth,
			plotY + plotHeight - normalized * plotHeight);
	};

	for (std::size_t i = 1; i < trainingPlan.steps.size(); ++i) {
		const auto a = mapPoint(i - 1, trainingPlan.steps[i - 1].discriminatorLoss);
		const auto b = mapPoint(i, trainingPlan.steps[i].discriminatorLoss);
		ofSetColor(245, 176, 65);
		ofDrawLine(a.x, a.y, b.x, b.y);
		const auto c = mapPoint(i - 1, trainingPlan.steps[i - 1].generatorLoss);
		const auto d = mapPoint(i, trainingPlan.steps[i].generatorLoss);
		ofSetColor(91, 192, 222);
		ofDrawLine(c.x, c.y, d.x, d.y);
	}
	for (std::size_t i = 0; i < trainingPlan.steps.size(); ++i) {
		const auto d = mapPoint(i, trainingPlan.steps[i].discriminatorLoss);
		ofSetColor(245, 176, 65);
		ofDrawCircle(d.x, d.y, 2.5f);
		const auto g = mapPoint(i, trainingPlan.steps[i].generatorLoss);
		ofSetColor(91, 192, 222);
		ofDrawCircle(g.x, g.y, 2.5f);
	}

	ofSetColor(245, 176, 65);
	ofDrawBitmapString("D", plotX + 8.0f, plotY + 16.0f);
	ofSetColor(91, 192, 222);
	ofDrawBitmapString("G", plotX + 28.0f, plotY + 16.0f);
}

#include "ofApp.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

namespace {
	const char* generatorModeNames[] = {
		"Built-in",
		"Preset",
		"Preview preset"
	};

	std::string firstLine(const std::string& text) {
		const auto lineBreak = text.find('\n');
		if (lineBreak == std::string::npos) {
			return text;
		}
		return text.substr(0, lineBreak);
	}
}

void ofApp::setup() {
	ofSetWindowTitle("ofxGgmlDiffusion GAN example");
	gui.setup();

	prompt = "small monochrome icon set for an openFrameworks tool";
	std::snprintf(promptBuffer.data(), promptBuffer.size(), "%s", prompt.c_str());

	backend = std::make_unique<ofxGgmlDiffusionTinyGanBackend>();
	rebuildRequest();
	updateTrainingPlan();
	refreshFixtureTextures();

	status = ofxGgmlDiffusionUtils::describe(request);
	if (backend->isAvailable()) {
		detail = "Ready";
	} else {
		detail = "Run ofxGgmlCore scripts\\setup-ggml.bat, then rebuild this example";
	}
	ofLogNotice("ofxGgmlDiffusionGanExample") << status;
	ofLogNotice("ofxGgmlDiffusionGanExample") << detail;
}

void ofApp::update() {
}

void ofApp::keyPressed(int key) {
	if (key == 'r' || key == 'R') {
		runGeneration();
	}
}

void ofApp::runGeneration() {
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

	if (ImGui::Combo("Generator", &generatorModeIndex, generatorModeNames, 3)) {
		generatorMode = static_cast<GeneratorMode>(generatorModeIndex);
		rebuildRequest();
	}

	if (ImGui::Button("Run")) {
		runGeneration();
	}
	ImGui::SameLine();
	if (ImGui::Button("Refresh plan")) {
		updateTrainingPlan();
		refreshFixtureTextures();
	}

	ImGui::Separator();
	ImGui::Text("Status: %s", status.c_str());
	ImGui::TextWrapped("%s", detail.c_str());
	ImGui::TextWrapped("Generator path: %s", request.gan.generatorPath.c_str());

	ImGui::Separator();
	bool settingsChanged = false;
	settingsChanged |= ImGui::SliderInt("Fixtures", &fixtureCount, 1, 32);
	settingsChanged |= ImGui::SliderInt("Epochs", &epochs, 1, 16);
	settingsChanged |= ImGui::SliderInt("Batch size", &batchSize, 1, 32);
	settingsChanged |= ImGui::SliderInt("Dry-run batches", &dryRunBatchesPerEpoch, 1, 16);
	settingsChanged |= ImGui::SliderFloat("Learning rate", &learningRate, 0.0001f, 0.01f, "%.4f");
	if (settingsChanged) {
		updateTrainingPlan();
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

	ImGui::End();
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
	request = ofxGgmlDiffusionUtils::makeGanImageRequest(prompt, findGeneratorPath());
	request.width = 512;
	request.height = 512;
	request.outputPath = getOutputPath();
	request.seed = 1234;
	request.gan.latentSize = 512;
	request.gan.truncation = 0.85f;
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
	detail = "Saved " + result.outputPath;
	ofLogNotice("ofxGgmlDiffusionGanExample") << detail;
}

std::string ofApp::findGeneratorPath() const {
	if (generatorMode == GeneratorMode::Preset) {
		return getPresetPath();
	}
	if (generatorMode == GeneratorMode::PreviewPreset) {
		return getPreviewPresetPath();
	}
	return "builtin:tiny-mlp";
}

std::string ofApp::getPresetPath() const {
	const char* envGenerator = std::getenv("OFXGGML_GAN_GENERATOR");
	if (envGenerator && ofFile::doesFileExist(envGenerator, false)) {
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

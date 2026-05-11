#pragma once

#include "ofMain.h"
#include "ofxGgmlDiffusion.h"
#include "ofxImGui.h"

#include <array>
#include <memory>
#include <string>

class ofApp : public ofBaseApp {
public:
	void setup() override;
	void update() override;
	void keyPressed(int key) override;
	void draw() override;

private:
	enum class GeneratorMode {
		Builtin = 0,
		Preset,
		PreviewPreset
	};

	void runGeneration();
	void drawGui();
	void createFixtureDataset();
	void writePreviewPreset();
	void updateTrainingPlan();
	std::string findGeneratorPath() const;
	std::string getPresetPath() const;
	std::string getPreviewPresetPath() const;
	std::string getPreviewDatasetPath() const;
	std::string getOutputPath() const;
	void rebuildRequest();
	void applyResult(ofxGgmlDiffusionResult result);

	ofxImGui::Gui gui;
	ofxGgmlDiffusionContextSettings settings;
	ofxGgmlDiffusionRequest request;
	std::unique_ptr<ofxGgmlDiffusionImageGenerationBackend> backend;
	ofTexture texture;
	GeneratorMode generatorMode = GeneratorMode::Builtin;
	int generatorModeIndex = 0;
	int fixtureCount = 8;
	int epochs = 2;
	int batchSize = 4;
	int dryRunBatchesPerEpoch = 3;
	float learningRate = 0.001f;
	ofxGgmlDiffusionTinyGanTrainingResult trainingPlan;
	std::array<char, 512> promptBuffer{};
	std::string prompt;
	std::string status;
	std::string detail;
	std::string trainingSummary;
};

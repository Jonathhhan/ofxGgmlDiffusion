#pragma once

#include "ofMain.h"
#if __has_include("ofxGgmlDiffusion.h")
#include "ofxGgmlDiffusion.h"
#elif __has_include("../../src/ofxGgmlDiffusion.h")
#include "../../src/ofxGgmlDiffusion.h"
#else
#error "Cannot find ofxGgmlDiffusion.h. Ensure the ofxGgmlDiffusion addon is linked correctly."
#endif
#if __has_include("ofxImGui.h")
#include "ofxImGui.h"
#elif __has_include("../../../ofxImGui/src/ofxImGui.h")
#include "../../../ofxImGui/src/ofxImGui.h"
#else
#error "Cannot find ofxImGui.h. Ensure the ofxImGui addon is linked correctly."
#endif

#include <array>
#include <memory>
#include <string>
#include <vector>

class ofApp : public ofBaseApp {
public:
	void setup() override;
	void update() override;
	void keyPressed(int key) override;
	void draw() override;

private:
	enum class GeneratorBackendMode {
		Proof = 0,
		Production
	};

	enum class GeneratorMode {
		Builtin = 0,
		Preset,
		PreviewPreset
	};

	void rebuildBackend();
	void runGeneration();
	void drawGui();
	void createFixtureDataset();
	void drawProofControls();
	void drawProductionControls();
	void drawStatus();
	void writePreviewPreset();
	void updateTrainingPlan();
	void refreshFixtureTextures();
	void drawTrainingPreview(float x, float y, float width, float height);
	void drawLossCurve(float x, float y, float width, float height);
	std::string findGeneratorPath() const;
	std::string findProductionGeneratorPath() const;
	std::string getPresetPath() const;
	std::string getPreviewPresetPath() const;
	std::string getPreviewDatasetPath() const;
	std::string getOutputPath() const;
	void rebuildRequest();
	void refreshProductionPathBuffer();
	void applyResult(ofxGgmlDiffusionResult result);

	ofxImGui::Gui gui;
	ofxGgmlDiffusionContextSettings settings;
	ofxGgmlDiffusionRequest request;
	std::unique_ptr<ofxGgmlDiffusionImageGenerationBackend> backend;
	ofTexture texture;
	std::vector<ofTexture> fixtureTextures;
	GeneratorBackendMode backendMode = GeneratorBackendMode::Proof;
	GeneratorMode generatorMode = GeneratorMode::Builtin;
	std::array<char, 768> productionGeneratorPathBuffer{};
	int backendModeIndex = 0;
	std::string productionGeneratorPath;
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

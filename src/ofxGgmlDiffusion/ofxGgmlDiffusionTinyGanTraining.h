#pragma once

#include <string>
#include <vector>

struct ofxGgmlDiffusionTinyGanTrainingSettings {
	std::string datasetPath;
	std::string outputPresetPath;
	int imageWidth = 64;
	int imageHeight = 64;
	int latentSize = 512;
	int hiddenSize = 96;
	int discriminatorHiddenSize = 96;
	int epochs = 1;
	int batchSize = 4;
	int dryRunBatchesPerEpoch = 1;
	float learningRate = 0.001f;
	bool dryRun = true;
};

struct ofxGgmlDiffusionTinyGanTrainingStep {
	int epoch = 0;
	int batch = 0;
	float discriminatorLoss = 0.0f;
	float generatorLoss = 0.0f;
};

struct ofxGgmlDiffusionTinyGanDatasetScan {
	std::string path;
	bool exists = false;
	bool isDirectory = false;
	int imageCount = 0;
	int unsupportedFileCount = 0;
	std::vector<std::string> imagePaths;
	std::vector<std::string> warnings;
};

struct ofxGgmlDiffusionTinyGanTrainingResult {
	bool success = false;
	std::string text;
	std::string error;
	std::string outputPresetPath;
	int epochsPlanned = 0;
	int batchSize = 0;
	int batchesPerEpoch = 0;
	int plannedDiscriminatorUpdates = 0;
	int plannedGeneratorUpdates = 0;
	float learningRate = 0.0f;
	float finalDiscriminatorLoss = 0.0f;
	float finalGeneratorLoss = 0.0f;
	ofxGgmlDiffusionTinyGanDatasetScan dataset;
	std::vector<std::string> warnings;
	std::vector<ofxGgmlDiffusionTinyGanTrainingStep> steps;

	bool isOk() const {
		return success;
	}

	bool isError() const {
		return !success;
	}

	explicit operator bool() const {
		return success;
	}
};

ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionValidateTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings);
ofxGgmlDiffusionTinyGanDatasetScan ofxGgmlDiffusionScanTinyGanDataset(
	const std::string& datasetPath);
std::vector<ofxGgmlDiffusionTinyGanTrainingStep> ofxGgmlDiffusionBuildTinyGanTrainingDryRunSteps(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings);
ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionPlanTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings);
ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionRunTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings);

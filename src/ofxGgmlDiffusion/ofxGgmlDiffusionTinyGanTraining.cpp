#include "ofxGgmlDiffusionTinyGanTraining.h"

#include <cmath>
#include <sstream>

namespace {
	ofxGgmlDiffusionTinyGanTrainingResult makeTrainingError(const std::string& message) {
		ofxGgmlDiffusionTinyGanTrainingResult result;
		result.success = false;
		result.error = message;
		return result;
	}

	void appendError(std::string& errors, const std::string& message) {
		if (!errors.empty()) {
			errors += "; ";
		}
		errors += message;
	}

	bool isFinitePositive(float value) {
		return std::isfinite(value) && value > 0.0f;
	}
}

ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionValidateTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings) {
	std::string errors;
	if (settings.imageWidth != 64 || settings.imageHeight != 64) {
		appendError(errors, "tiny GAN training currently supports 64x64 images only");
	}
	if (settings.latentSize < 8 || settings.latentSize > 1024) {
		appendError(errors, "latentSize must be between 8 and 1024");
	}
	if (settings.hiddenSize < 8 || settings.hiddenSize > 512) {
		appendError(errors, "hiddenSize must be between 8 and 512");
	}
	if (settings.epochs < 1) {
		appendError(errors, "epochs must be at least 1");
	}
	if (settings.batchSize < 1) {
		appendError(errors, "batchSize must be at least 1");
	}
	if (!isFinitePositive(settings.learningRate)) {
		appendError(errors, "learningRate must be finite and greater than zero");
	}
	if (settings.outputPresetPath.empty()) {
		appendError(errors, "outputPresetPath is required");
	}
	if (!settings.dryRun) {
		appendError(errors, "tiny GAN training currently supports dry-run planning only");
	}

	if (!errors.empty()) {
		return makeTrainingError(errors);
	}

	ofxGgmlDiffusionTinyGanTrainingResult result;
	result.success = true;
	result.text = "tiny GAN training settings are valid";
	result.outputPresetPath = settings.outputPresetPath;
	result.epochsPlanned = settings.epochs;
	result.batchSize = settings.batchSize;
	result.learningRate = settings.learningRate;
	if (settings.datasetPath.empty()) {
		result.warnings.push_back("no dataset path provided; planning uses a synthetic dataset placeholder");
	}
	return result;
}

ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionPlanTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings) {
	auto result = ofxGgmlDiffusionValidateTinyGanTraining(settings);
	if (!result) {
		return result;
	}

	std::ostringstream text;
	text << "tiny GAN training dry-run\n";
	text << "dataset: " << (settings.datasetPath.empty() ? "(synthetic placeholder)" : settings.datasetPath) << "\n";
	text << "output preset: " << settings.outputPresetPath << "\n";
	text << "architecture: tiny-mlp\n";
	text << "image size: " << settings.imageWidth << "x" << settings.imageHeight << "\n";
	text << "latent size: " << settings.latentSize << "\n";
	text << "hidden size: " << settings.hiddenSize << "\n";
	text << "epochs: " << settings.epochs << "\n";
	text << "batch size: " << settings.batchSize << "\n";
	text << "learning rate: " << settings.learningRate << "\n";
	text << "status: validated planning only; no weights were trained";
	result.text = text.str();
	return result;
}

ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionRunTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings) {
	if (!settings.dryRun) {
		return makeTrainingError(
			"tiny GAN training is experimental and currently supports dry-run planning only");
	}
	return ofxGgmlDiffusionPlanTinyGanTraining(settings);
}

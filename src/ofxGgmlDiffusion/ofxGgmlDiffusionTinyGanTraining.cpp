#include "ofxGgmlDiffusionTinyGanTraining.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
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

	std::string toLower(std::string value) {
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return value;
	}

	bool isSupportedImageExtension(const std::filesystem::path& path) {
		const auto extension = toLower(path.extension().string());
		return extension == ".png" ||
			extension == ".jpg" ||
			extension == ".jpeg" ||
			extension == ".bmp" ||
			extension == ".tga" ||
			extension == ".ppm";
	}

	int fixtureChannelValue(int x, int y, int index, int channel) {
		const int stripe = ((x / 8) + index + channel) % 2;
		const int checker = ((x / 8) + (y / 8) + index) % 2;
		const int gradient = (x * 3 + y * 5 + index * 29 + channel * 47) % 256;
		if (index % 3 == 0) {
			return stripe ? 224 : 32;
		}
		if (index % 3 == 1) {
			return checker ? 196 : 56;
		}
		return gradient;
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
	if (settings.discriminatorHiddenSize < 8 || settings.discriminatorHiddenSize > 512) {
		appendError(errors, "discriminatorHiddenSize must be between 8 and 512");
	}
	if (settings.epochs < 1) {
		appendError(errors, "epochs must be at least 1");
	}
	if (settings.batchSize < 1) {
		appendError(errors, "batchSize must be at least 1");
	}
	if (settings.dryRunBatchesPerEpoch < 1 || settings.dryRunBatchesPerEpoch > 128) {
		appendError(errors, "dryRunBatchesPerEpoch must be between 1 and 128");
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
	result.batchesPerEpoch = settings.dryRunBatchesPerEpoch;
	result.plannedDiscriminatorUpdates = settings.epochs * settings.dryRunBatchesPerEpoch;
	result.plannedGeneratorUpdates = settings.epochs * settings.dryRunBatchesPerEpoch;
	result.learningRate = settings.learningRate;
	if (settings.datasetPath.empty()) {
		result.warnings.push_back("no dataset path provided; planning uses a synthetic dataset placeholder");
	}
	return result;
}

ofxGgmlDiffusionTinyGanDatasetScan ofxGgmlDiffusionScanTinyGanDataset(
	const std::string& datasetPath) {
	ofxGgmlDiffusionTinyGanDatasetScan scan;
	scan.path = datasetPath;
	if (datasetPath.empty()) {
		scan.warnings.push_back("no dataset path provided; planning uses a synthetic dataset placeholder");
		return scan;
	}

	const std::filesystem::path root(datasetPath);
	std::error_code error;
	scan.exists = std::filesystem::exists(root, error);
	if (error || !scan.exists) {
		scan.warnings.push_back("dataset path does not exist");
		return scan;
	}

	scan.isDirectory = std::filesystem::is_directory(root, error);
	if (error || !scan.isDirectory) {
		scan.warnings.push_back("dataset path is not a directory");
		return scan;
	}

	for (const auto& entry : std::filesystem::recursive_directory_iterator(root, error)) {
		if (error) {
			scan.warnings.push_back("dataset scan stopped early because a directory could not be read");
			break;
		}
		if (!entry.is_regular_file(error) || error) {
			error.clear();
			continue;
		}
		if (isSupportedImageExtension(entry.path())) {
			++scan.imageCount;
			scan.imagePaths.push_back(entry.path().string());
		} else {
			++scan.unsupportedFileCount;
		}
	}

	std::sort(scan.imagePaths.begin(), scan.imagePaths.end());
	if (scan.imageCount == 0) {
		scan.warnings.push_back("dataset contains no supported images");
	}
	return scan;
}

bool ofxGgmlDiffusionWriteTinyGanFixtureDataset(
	const std::string& datasetPath,
	int imageCount,
	std::string& error) {
	error.clear();
	if (datasetPath.empty()) {
		error = "datasetPath is required";
		return false;
	}
	if (imageCount < 1 || imageCount > 1024) {
		error = "imageCount must be between 1 and 1024";
		return false;
	}

	const std::filesystem::path root(datasetPath);
	std::error_code code;
	std::filesystem::create_directories(root, code);
	if (code) {
		error = "could not create fixture dataset directory";
		return false;
	}

	for (int index = 0; index < imageCount; ++index) {
		std::ostringstream name;
		name << "fixture-" << std::setw(3) << std::setfill('0') << index << ".ppm";
		const auto path = root / name.str();
		std::ofstream file(path, std::ios::out | std::ios::trunc);
		if (!file) {
			error = "could not write fixture image: " + path.string();
			return false;
		}

		file << "P3\n64 64\n255\n";
		for (int y = 0; y < 64; ++y) {
			for (int x = 0; x < 64; ++x) {
				file << fixtureChannelValue(x, y, index, 0) << " "
					<< fixtureChannelValue(x, y, index, 1) << " "
					<< fixtureChannelValue(x, y, index, 2);
				if (x < 63) {
					file << " ";
				}
			}
			file << "\n";
		}
	}

	return true;
}

std::vector<ofxGgmlDiffusionTinyGanTrainingStep> ofxGgmlDiffusionBuildTinyGanTrainingDryRunSteps(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings) {
	std::vector<ofxGgmlDiffusionTinyGanTrainingStep> steps;
	const int totalSteps = settings.epochs * settings.dryRunBatchesPerEpoch;
	if (totalSteps <= 0) {
		return steps;
	}

	steps.reserve(static_cast<std::size_t>(totalSteps));
	for (int epoch = 0; epoch < settings.epochs; ++epoch) {
		for (int batch = 0; batch < settings.dryRunBatchesPerEpoch; ++batch) {
			const int index = epoch * settings.dryRunBatchesPerEpoch + batch;
			const float progress = static_cast<float>(index + 1) / static_cast<float>(totalSteps);
			ofxGgmlDiffusionTinyGanTrainingStep step;
			step.epoch = epoch + 1;
			step.batch = batch + 1;
			step.discriminatorLoss = 1.3862944f - (0.18f * progress);
			step.generatorLoss = 0.6931472f - (0.07f * progress);
			steps.push_back(step);
		}
	}
	return steps;
}

ofxGgmlDiffusionTinyGanTrainingResult ofxGgmlDiffusionPlanTinyGanTraining(
	const ofxGgmlDiffusionTinyGanTrainingSettings& settings) {
	auto result = ofxGgmlDiffusionValidateTinyGanTraining(settings);
	if (!result) {
		return result;
	}
	result.dataset = ofxGgmlDiffusionScanTinyGanDataset(settings.datasetPath);
	for (const auto& warning : result.dataset.warnings) {
		result.warnings.push_back(warning);
	}
	result.steps = ofxGgmlDiffusionBuildTinyGanTrainingDryRunSteps(settings);
	if (!result.steps.empty()) {
		result.finalDiscriminatorLoss = result.steps.back().discriminatorLoss;
		result.finalGeneratorLoss = result.steps.back().generatorLoss;
	}

	std::ostringstream text;
	text << "tiny GAN training dry-run\n";
	text << "dataset: " << (settings.datasetPath.empty() ? "(synthetic placeholder)" : settings.datasetPath) << "\n";
	text << "dataset exists: " << (result.dataset.exists ? "yes" : "no") << "\n";
	text << "dataset images: " << result.dataset.imageCount << "\n";
	text << "dataset unsupported files: " << result.dataset.unsupportedFileCount << "\n";
	text << "output preset: " << settings.outputPresetPath << "\n";
	text << "generator architecture: tiny-mlp\n";
	text << "discriminator architecture: tiny-mlp-binary-classifier\n";
	text << "image size: " << settings.imageWidth << "x" << settings.imageHeight << "\n";
	text << "latent size: " << settings.latentSize << "\n";
	text << "generator hidden size: " << settings.hiddenSize << "\n";
	text << "discriminator hidden size: " << settings.discriminatorHiddenSize << "\n";
	text << "epochs: " << settings.epochs << "\n";
	text << "batch size: " << settings.batchSize << "\n";
	text << "dry-run batches per epoch: " << settings.dryRunBatchesPerEpoch << "\n";
	text << "planned discriminator updates: " << result.plannedDiscriminatorUpdates << "\n";
	text << "planned generator updates: " << result.plannedGeneratorUpdates << "\n";
	text << "learning rate: " << settings.learningRate << "\n";
	text << "final dry-run discriminator loss: " << result.finalDiscriminatorLoss << "\n";
	text << "final dry-run generator loss: " << result.finalGeneratorLoss << "\n";
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

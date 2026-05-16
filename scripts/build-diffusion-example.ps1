param(
	[string]$Configuration = "Release",
	[string]$Platform = "x64",
	[string]$Example = "ofxGgmlDiffusionPromptExample",
	[int]$Jobs = 1,
	[switch]$Clean
)

$ErrorActionPreference = "Stop"

function Write-Step {
	param([string]$Message)
	Write-Host "==> $Message"
}

function Test-WindowsHost {
	return !($IsLinux -or $IsMacOS)
}

function Invoke-CheckedNative {
	param(
		[string]$Step,
		[scriptblock]$Command
	)
	& $Command
	if ($LASTEXITCODE -ne 0) {
		throw "$Step failed with exit code $LASTEXITCODE"
	}
}

function Get-StableNameFragment {
	param([string]$Text)
	$sha1 = [System.Security.Cryptography.SHA1]::Create()
	try {
		$bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
		$hash = $sha1.ComputeHash($bytes)
		return [System.BitConverter]::ToString($hash).Replace("-", "")
	} finally {
		$sha1.Dispose()
	}
}

function Invoke-WithNamedMutex {
	param(
		[string]$Name,
		[scriptblock]$Command
	)
	$mutex = New-Object System.Threading.Mutex($false, $Name)
	$locked = $false
	try {
		$locked = $mutex.WaitOne([TimeSpan]::FromMinutes(30))
		if (!$locked) {
			throw "Timed out waiting for build lock: $Name"
		}
		& $Command
	} finally {
		if ($locked) {
			$mutex.ReleaseMutex()
		}
		$mutex.Dispose()
	}
}

function Get-MsBuild {
	$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
	if (Test-Path -LiteralPath $vswhere) {
		$installPath = & $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath 2>$null
		if ($installPath) {
			$candidate = Join-Path $installPath "MSBuild\Current\Bin\MSBuild.exe"
			if (Test-Path -LiteralPath $candidate) {
				return $candidate
			}
		}
	}

	foreach ($version in @("18", "17", "16")) {
		foreach ($edition in @("Community", "Professional", "Enterprise", "BuildTools")) {
			$candidate = "C:\Program Files\Microsoft Visual Studio\$version\$edition\MSBuild\Current\Bin\MSBuild.exe"
			if (Test-Path -LiteralPath $candidate) {
				return $candidate
			}
		}
	}
	return ""
}

function Resolve-BuildJobs {
	param([int]$RequestedJobs)
	if ($RequestedJobs -lt 0) {
		throw "-Jobs must be 0 or greater."
	}
	if ($RequestedJobs -eq 0) {
		return [Environment]::ProcessorCount
	}
	return $RequestedJobs
}

function Get-MsBuildParallelArguments {
	param([int]$BuildJobs)
	if ($BuildJobs -gt 1) {
		return @("/p:MultiProcessorCompilation=true", "/m:$BuildJobs")
	}
	return @("/p:MultiProcessorCompilation=false", "/m:1")
}

function Get-RelativeProjectPath {
	param(
		[string]$ProjectDir,
		[string]$FilePath
	)
	$projectUri = [System.Uri]((Resolve-Path -LiteralPath $ProjectDir).Path.TrimEnd("\") + "\")
	$fileUri = [System.Uri](Resolve-Path -LiteralPath $FilePath).Path
	return [System.Uri]::UnescapeDataString(
		$projectUri.MakeRelativeUri($fileUri).ToString()).Replace("/", "\")
}

function Add-VisualStudioProjectItem {
	param(
		[xml]$Doc,
		[System.Xml.XmlNamespaceManager]$Namespace,
		[string]$Tag,
		[string]$Include
	)
	$existing = $Doc.SelectSingleNode("//msb:$Tag[@Include='$Include']", $Namespace)
	if ($existing) {
		return $false
	}
	$itemGroups = @($Doc.SelectNodes("//msb:ItemGroup", $Namespace))
	$itemGroup = $null
	foreach ($group in $itemGroups) {
		if ($group.SelectSingleNode("msb:$Tag", $Namespace)) {
			$itemGroup = $group
			break
		}
	}
	if (!$itemGroup -and $itemGroups.Count -gt 0) {
		$itemGroup = $itemGroups[0]
	}
	if (!$itemGroup) {
		return $false
	}
	$item = $Doc.CreateElement($Tag, $Doc.DocumentElement.NamespaceURI)
	$item.SetAttribute("Include", $Include)
	[void]$itemGroup.AppendChild($item)
	return $true
}

function Repair-DiffusionGeneratedProject {
	param(
		[string]$Project,
		[string]$AddonRoot
	)
	if (!(Test-Path -LiteralPath $Project -PathType Leaf)) {
		return
	}

	[xml]$doc = Get-Content -LiteralPath $Project -Raw
	$namespace = New-Object System.Xml.XmlNamespaceManager($doc.NameTable)
	$namespace.AddNamespace("msb", "http://schemas.microsoft.com/developer/msbuild/2003")
	$projectDir = Split-Path -Parent $Project
	$changed = $false

	$includeDirs = @(
		"..\src",
		"..\libs\stable-diffusion\include"
	)
	foreach ($node in @($doc.SelectNodes("//msb:AdditionalIncludeDirectories", $namespace))) {
		$parts = @($node.InnerText -split ";" | Where-Object { $_ })
		foreach ($includeDir in $includeDirs) {
			if ($parts -notcontains $includeDir) {
				$parts += $includeDir
				$changed = $true
			}
		}
		$node.InnerText = $parts -join ";"
	}

	$sourceRoot = Join-Path $AddonRoot "src"
	Get-ChildItem -LiteralPath $sourceRoot -Recurse -File | ForEach-Object {
		$relative = Get-RelativeProjectPath -ProjectDir $projectDir -FilePath $_.FullName
		if ($_.Extension -in @(".cpp", ".cxx", ".cc")) {
			if (Add-VisualStudioProjectItem -Doc $doc -Namespace $namespace -Tag "ClCompile" -Include $relative) {
				$changed = $true
			}
		} elseif ($_.Extension -in @(".h", ".hpp")) {
			if (Add-VisualStudioProjectItem -Doc $doc -Namespace $namespace -Tag "ClInclude" -Include $relative) {
				$changed = $true
			}
		}
	}

	foreach ($node in @($doc.SelectNodes("//msb:PostBuildEvent/msb:Command", $namespace))) {
		if ($node.InnerText -match '\$\(ProjectDir\)dll\\([^\\]+)\\\*\.dll') {
			$platformName = $matches[1]
			$guardedCommand = "if exist `"`$(ProjectDir)dll\$platformName\*.dll`" xcopy /Y /E `"`$(ProjectDir)dll\$platformName\*.dll`" `"`$(TargetDir)`""
			if ($node.InnerText -ne $guardedCommand) {
				$node.InnerText = $guardedCommand
				$changed = $true
			}
		}
	}

	if ($changed) {
		$doc.Save($Project)
		Write-Step "Repaired generated Visual Studio project metadata"
	}
}

function Initialize-DiffusionGeneratedProject {
	param(
		[string]$ExampleRoot,
		[string]$Example,
		[string]$OfRoot
	)
	$templateRoot = Join-Path $OfRoot "scripts\templates\winvs"
	$templateProject = Join-Path $templateRoot "emptyExample.vcxproj"
	if (!(Test-Path -LiteralPath $templateProject -PathType Leaf)) {
		throw "Visual Studio project not found and openFrameworks winvs template is missing: $templateProject"
	}
	$copies = @(
		@{ Source = "emptyExample.vcxproj"; Target = "$Example.vcxproj" },
		@{ Source = "emptyExample.vcxproj.filters"; Target = "$Example.vcxproj.filters" },
		@{ Source = "emptyExample.vcxproj.user"; Target = "$Example.vcxproj.user" },
		@{ Source = "emptyExample.sln"; Target = "$Example.sln" },
		@{ Source = "icon.rc"; Target = "icon.rc" }
	)
	foreach ($copy in $copies) {
		$source = Join-Path $templateRoot $copy.Source
		$target = Join-Path $ExampleRoot $copy.Target
		if (!(Test-Path -LiteralPath $source -PathType Leaf)) {
			continue
		}
		$text = Get-Content -LiteralPath $source -Raw
		$text = $text.Replace("emptyExample", $Example)
		Set-Content -LiteralPath $target -Value $text -NoNewline
	}
	Write-Step "Initialized generated Visual Studio project metadata from openFrameworks template"
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$addonRoot = Resolve-Path (Join-Path $scriptRoot "..")
$ofRoot = Resolve-Path (Join-Path $addonRoot "..\..")
$exampleRoot = Join-Path $addonRoot $Example
if (!(Test-Path -LiteralPath $exampleRoot -PathType Container)) {
	throw "Example directory not found: $exampleRoot"
}

if (Test-WindowsHost) {
	$project = Join-Path $exampleRoot "$Example.vcxproj"
	if (!(Test-Path -LiteralPath $project -PathType Leaf)) {
		Initialize-DiffusionGeneratedProject -ExampleRoot $exampleRoot -Example $Example -OfRoot $ofRoot
	}
	Repair-DiffusionGeneratedProject -Project $project -AddonRoot $addonRoot
	$msbuild = Get-MsBuild
	if ([string]::IsNullOrWhiteSpace($msbuild)) {
		throw "MSBuild.exe was not found."
	}

	$target = if ($Clean) { "Rebuild" } else { "Build" }
	$buildJobs = Resolve-BuildJobs -RequestedJobs $Jobs
	$parallelArgs = Get-MsBuildParallelArguments -BuildJobs $buildJobs
	Write-Step "Building $Example $Configuration $Platform with MSBuild ($buildJobs jobs)"
	$lockName = "Local\ofxGgml-msbuild-" + (Get-StableNameFragment $ofRoot.Path)
	Invoke-WithNamedMutex -Name $lockName -Command {
		& $msbuild $project /t:$target /p:Configuration=$Configuration /p:Platform=$Platform @parallelArgs /nr:false
		if ($LASTEXITCODE -ne 0) {
			throw "MSBuild $Example failed with exit code $LASTEXITCODE"
		}
	}
	return
}

$makefile = Join-Path $exampleRoot "Makefile"
if (Test-Path -LiteralPath $makefile -PathType Leaf) {
	$target = if ($Clean) { "clean Release" } else { "Release" }
	Write-Step "Building $Example with make"
	Invoke-CheckedNative "make $Example" {
		make -C $exampleRoot $target
	}
	return
}

if ($IsMacOS) {
	$xcodeProject = Get-ChildItem -LiteralPath $exampleRoot -Filter "*.xcodeproj" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
	if ($xcodeProject) {
		Write-Step "Building $Example $Configuration with xcodebuild"
		Invoke-CheckedNative "xcodebuild $Example" {
			xcodebuild -project $xcodeProject.FullName -configuration $Configuration
		}
		return
	}
}

throw "No supported generated project was found for $Example. Generate the example project with openFrameworks projectGenerator first."

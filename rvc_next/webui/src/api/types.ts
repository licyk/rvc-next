import type { components } from '@/api/schema';

/** Shorthand for the generated schemas. */
export type S = components['schemas'];

// One schema per model (separate_input_output_schemas=False), except the training models below,
// which FastAPI still splits; the client reads their output shape and sends the input one.
export type VoiceParams = S['VoiceParamsModel'];
export type VoiceParamsInput = S['VoiceParamsModel'];
export type StreamParams = S['StreamParamsModel'];
export type LiveDevices = S['LiveDevices'];
export type DeviceSelection = S['DeviceSelection'];
export type LiveConfig = S['LiveConfig'];
export type Dataset = S['Dataset-Output'];
export type FitSettings = S['FitSettings-Output'];
export type SpeakerEntry = S['SpeakerEntry-Output'];

export type AppMeta = S['AppMeta'];
export type SettingsView = S['SettingsView'];
export type Job = S['Job'];
export type JobStep = S['JobStep'];
export type JobPage = S['JobPage'];
export type JobLog = S['JobLog'];
export type Output = S['Output'];
export type OutputPage = S['OutputPage'];
export type AudioFile = S['AudioFile'];
export type AudioRef = S['AudioRef'];
export type Peaks = S['Peaks'];
export type BrowseListing = S['BrowseListing'];
export type BrowseEntry = S['BrowseEntry'];
export type VoiceModel = S['VoiceModel'];
export type Speaker = S['Speaker'];
export type Preset = S['Preset'];
export type AssetStatus = S['AssetStatus'];
export type SeparationPreset = S['SeparationPreset'];
export type ComputeUsage = S['ComputeUsage'];
export type ComputeInfo = S['ComputeInfo'];
export type ConvertRequest = S['ConvertRequest'];
export type Experiment = S['Experiment'];
export type ExperimentSummary = S['ExperimentSummary'];
export type StageState = S['StageState'];
export type DatasetReport = S['DatasetReport'];
export type Checkpoint = S['Checkpoint'];
export type TrainMetric = S['TrainMetric'];
export type LiveState = S['LiveState'];
// Event schemas mark defaulted fields optional; every field is always sent.
export type LiveStats = S['LiveStats'];
export type DeviceList = S['DeviceList'];
export type PhysicalDevice = S['PhysicalDevice'];
export type AudioDevice = S['AudioDevice'];
export type DeviceCheck = S['DeviceCheck'];
export type LatencyMeasurement = S['LatencyMeasurement'];
export type ResolvedDevice = S['ResolvedDevice'];
export type ImportResult = S['ImportResult'];

export type JobState = Job['state'];
export type LiveStateName = LiveState['state'];

/** Socket event payloads, typed from the same schema as REST. */
export type ServerEvents = { [K in keyof S['ServerEvents']]: S['ServerEvents'][K] };
export type CatalogVoice = S['CatalogVoice'];
export type Repository = S['Repository'];

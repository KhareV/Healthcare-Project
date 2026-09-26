// Shared Trajectory Copilot context — a small rune-based singleton so any
// page (Patient Replay, AI + SHAP, the guided demo) can open the drawer
// already attached to the patient/cutoff currently being viewed, without
// prop-drilling through DashboardShell. The drawer itself lives in
// DashboardShell.svelte; this module only holds *what* it should be
// grounded in right now.

export type CopilotContext = {
	open: boolean;
	stayId: string | null;
	patientAlias: string | null;
	predictionTime: string | null;
	previousPredictionTime: string | null;
};

export const copilotState = $state<CopilotContext>({
	open: false,
	stayId: null,
	patientAlias: null,
	predictionTime: null,
	previousPredictionTime: null
});

export function setCopilotContext(params: { stayId: string; patientAlias: string; predictionTime: string; previousPredictionTime?: string | null }) {
	copilotState.stayId = params.stayId;
	copilotState.patientAlias = params.patientAlias;
	copilotState.predictionTime = params.predictionTime;
	copilotState.previousPredictionTime = params.previousPredictionTime ?? null;
}

export function openCopilot(params?: { stayId: string; patientAlias: string; predictionTime: string; previousPredictionTime?: string | null }) {
	if (params) setCopilotContext(params);
	copilotState.open = true;
}

export function closeCopilot() {
	copilotState.open = false;
}

{{/*
Expand the name of the chart.
*/}}
{{- define "wekai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "wekai.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "wekai.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "wekai.labels" -}}
helm.sh/chart: {{ include "wekai.chart" . }}
{{ include "wekai.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "wekai.selectorLabels" -}}
app.kubernetes.io/name: {{ include "wekai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Results are always written to resultsMountPath; the only question these two
helpers answer is what backs that path.

wekai.results.ownClaim — non-empty ("true") when the chart should create and
own the results PVC. An existing resultsClaim always wins: the user manages
that volume's lifecycle, so provisioning a second one alongside it would be
silent waste. storeResults is the pre-existingClaim name for
createResultsClaim and is still honored, so values files written against
older chart versions keep their PVC instead of silently degrading to an
emptyDir that vanishes with the pod.
*/}}
{{- define "wekai.results.ownClaim" -}}
{{- if and (not .Values.resultsClaim) (or .Values.createResultsClaim .Values.storeResults) -}}
true
{{- end -}}
{{- end }}

{{/*
wekai.results.claimName — the PVC to mount at resultsMountPath. Empty means
no PVC at all, i.e. an ephemeral emptyDir.
*/}}
{{- define "wekai.results.claimName" -}}
{{- if .Values.resultsClaim -}}
{{- .Values.resultsClaim -}}
{{- else if include "wekai.results.ownClaim" . -}}
{{- printf "%s-results" (include "wekai.fullname" .) -}}
{{- end -}}
{{- end }}

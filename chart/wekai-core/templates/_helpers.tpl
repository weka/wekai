{{/*
Expand the name of the chart.
*/}}
{{- define "wekai-core.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "wekai-core.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "wekai-core.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "wekai-core.labels" -}}
helm.sh/chart: {{ include "wekai-core.chart" . }}
{{ include "wekai-core.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "wekai-core.selectorLabels" -}}
app.kubernetes.io/name: {{ include "wekai-core.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

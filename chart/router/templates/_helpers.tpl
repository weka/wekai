{{/*
Expand the name of the chart.
*/}}
{{- define "app.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "app.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "app.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "app.labels" -}}
helm.sh/chart: {{ include "app.chart" . }}
{{ include "app.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "app.selectorLabels" -}}
app.kubernetes.io/name: {{ include "app.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
app.endpointToken renders ONE endpoint. There is exactly one endpoint grammar,
used everywhere an endpoint can appear — `backends` and a route's `endpoints`
are the same list of the same things.

A URL is a string. A pod selector is a map, and it has to be a map because
--set splits values on commas: a label selector is comma-separated, so the
string form arrives truncated at its first label — silently, rendering fine and
failing at runtime.

  http://vllm-a:8000                           ->  unchanged
  {pods: {app: vllm, tier: prod}, port: http}  ->  pods:app=vllm,tier=prod:http

Map keys render sorted, so the flag is stable across renders; selector order is
insignificant to Kubernetes either way.
*/}}
{{- define "app.endpointToken" -}}
{{- if kindIs "string" . -}}
{{ . }}
{{- else if .pods -}}
{{- $sel := list -}}
{{- range $k, $v := .pods }}{{ $sel = append $sel (printf "%s=%s" $k (toString $v)) }}{{ end -}}
{{- $tok := printf "pods:%s" (join "," $sel) -}}
{{- if .port }}{{ $tok = printf "%s:%v" $tok .port }}{{ end -}}
{{ $tok }}
{{- else -}}
{{ fail "router: an endpoint is a URL string or a map with `pods:` — got neither" }}
{{- end -}}
{{- end -}}

{{/*
app.endpointList renders a pipe-separated endpoint list. Mixing a URL and a
selector in one list is what a migration looks like.
*/}}
{{- define "app.endpointList" -}}
{{- $out := list -}}
{{- range . }}{{ $out = append $out (include "app.endpointToken" .) }}{{ end -}}
{{ join "|" $out }}
{{- end -}}

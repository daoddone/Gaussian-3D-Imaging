import SwiftUI

/// Minimal settings for the in-app Transmit path: the Linux upload receiver's base URL + bearer
/// token. Persisted to UserDefaults under exactly the keys `UploadConfig` reads
/// (`upload_base_url`, `upload_token`). Until both are set, Transmit reports "set server URL +
/// token in Settings first" and captures go out via AirDrop → Files instead.
struct SettingsView: View {
    @AppStorage("upload_base_url") private var baseURL: String = ""
    @AppStorage("upload_token") private var token: String = ""
    // Capture tuning (read by CaptureTuning / KeyframeSelector at each recording). 0 -> built-in
    // default (360 frames / 120 s). Raised from the shipped 60/20 so the STRONG-CAPTURE branch
    // (docs/PIPELINE_RECOMMENDATION.md: 150-400 sharp frames) is reachable. Compute cost note: more
    // frames = longer SfM + reconstruction on the Linux box; benchmark on the first strong capture.
    @AppStorage("max_keyframes") private var maxKeyframes: Int = 360
    @AppStorage("budget_seconds") private var budgetSeconds: Double = 120
    @Environment(\.dismiss) private var dismiss

    private var configured: Bool { !baseURL.isEmpty && !token.isEmpty }

    var body: some View {
        NavigationStack {
            Form {
                Section("Upload server") {
                    TextField("Base URL (e.g. https://host)", text: $baseURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    SecureField("Token", text: $token)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Section("Capture length") {
                    Stepper("Max frames: \(maxKeyframes)", value: $maxKeyframes, in: 30...1000, step: 30)
                    Stepper("Time budget: \(Int(budgetSeconds)) s", value: $budgetSeconds, in: 10...300, step: 10)
                    Text("Strong-capture branch wants 150-400 sharp frames over a slow orbit. Longer captures take longer to process on the server.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Section {
                    Text(configured
                         ? "Configured. Transmit will POST the capture zip to \(baseURL)/upload."
                         : "Both fields are required for in-app Transmit. Without them, send captures via AirDrop → Files → your Mac → ssh.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }
}

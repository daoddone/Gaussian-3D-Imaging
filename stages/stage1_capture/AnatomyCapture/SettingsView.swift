import SwiftUI

/// Minimal settings for the in-app Transmit path: the Linux upload receiver's base URL + bearer
/// token. Persisted to UserDefaults under exactly the keys `UploadConfig` reads
/// (`upload_base_url`, `upload_token`). Until both are set, Transmit reports "set server URL +
/// token in Settings first" and captures go out via AirDrop → Files instead.
struct SettingsView: View {
    @AppStorage("upload_base_url") private var baseURL: String = ""
    @AppStorage("upload_token") private var token: String = ""
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

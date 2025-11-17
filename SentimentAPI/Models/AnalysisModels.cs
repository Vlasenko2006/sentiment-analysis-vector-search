using System.Text.Json.Serialization;

namespace SentimentAPI.Models;

/// <summary>
/// Request model for starting sentiment analysis
/// </summary>
public class AnalysisRequest
{
    /// <summary>
    /// URL of the webpage to analyze (optional if using demo mode)
    /// </summary>
    [JsonPropertyName("url")]
    public string? Url { get; set; }

    /// <summary>
    /// HTML content to analyze (alternative to URL)
    /// </summary>
    [JsonPropertyName("html_content")]
    public string? HtmlContent { get; set; }

    /// <summary>
    /// Email address(es) for report delivery (optional)
    /// Can be: single email, comma-separated emails, or list of emails
    /// Examples: "user@example.com" or "user1@example.com, user2@example.com"
    /// </summary>
    [JsonPropertyName("email")]
    public string? Email { get; set; }

    /// <summary>
    /// Alternative: List of email addresses for report delivery (optional)
    /// </summary>
    [JsonPropertyName("emails")]
    public List<string>? Emails { get; set; }

    /// <summary>
    /// Custom prompt for AI recommendations (optional)
    /// </summary>
    [JsonPropertyName("customPrompt")]
    public string? CustomPrompt { get; set; }

    /// <summary>
    /// Search method: 'keywords', 'urls', or 'demo' (default: 'demo')
    /// </summary>
    [JsonPropertyName("searchMethod")]
    public string? SearchMethod { get; set; } = "demo";
}

/// <summary>
/// Job status response
/// </summary>
public class JobStatus
{
    [JsonPropertyName("job_id")]
    public required string JobId { get; set; }

    [JsonPropertyName("status")]
    public required string Status { get; set; }

    [JsonPropertyName("progress")]
    public int Progress { get; set; }

    [JsonPropertyName("message")]
    public string? Message { get; set; }

    [JsonPropertyName("pdf_url")]
    public string? PdfUrl { get; set; }

    [JsonPropertyName("results_url")]
    public string? ResultsUrl { get; set; }

    [JsonPropertyName("error")]
    public string? Error { get; set; }

    [JsonPropertyName("created_at")]
    public string? CreatedAt { get; set; }
}

/// <summary>
/// Analysis results data
/// </summary>
public class AnalysisResults
{
    [JsonPropertyName("trends")]
    public Dictionary<string, object>? Trends { get; set; }

    [JsonPropertyName("positive_summary")]
    public Dictionary<string, object>? PositiveSummary { get; set; }

    [JsonPropertyName("negative_summary")]
    public Dictionary<string, object>? NegativeSummary { get; set; }

    [JsonPropertyName("neutral_summary")]
    public Dictionary<string, object>? NeutralSummary { get; set; }

    [JsonPropertyName("recommendations")]
    public Dictionary<string, object>? Recommendations { get; set; }

    [JsonPropertyName("statistics")]
    public Dictionary<string, object>? Statistics { get; set; }
}

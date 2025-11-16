using SentimentAPI.Models;
using System.Text.Json;

namespace SentimentAPI.Services;

public interface IPythonServiceClient
{
    Task<JobStatus> StartAnalysisAsync(AnalysisRequest request);
    Task<JobStatus?> GetJobStatusAsync(string jobId);
    Task<byte[]?> GetPdfReportAsync(string jobId);
    Task<AnalysisResults?> GetAnalysisDataAsync(string jobId);
    Task<JobStatus> UploadFileAsync(IFormFile file);
    Task<bool> CheckHealthAsync();
    Task<string> SendChatMessageAsync(string jobId, string question, bool includeHistory = true);
    Task<List<string>> GetChatSuggestionsAsync(string jobId);
    Task<bool> ClearChatHistoryAsync(string jobId);
}

public class PythonServiceClient : IPythonServiceClient
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<PythonServiceClient> _logger;

    public PythonServiceClient(HttpClient httpClient, ILogger<PythonServiceClient> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
    }

    public async Task<JobStatus> StartAnalysisAsync(AnalysisRequest request)
    {
        try
        {
            _logger.LogInformation("Calling Python service to start analysis");
            var response = await _httpClient.PostAsJsonAsync("/api/analyze", request);
            response.EnsureSuccessStatusCode();
            
            var result = await response.Content.ReadFromJsonAsync<JobStatus>();
            if (result == null)
            {
                throw new Exception("Failed to deserialize response from Python service");
            }
            
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calling Python service");
            throw;
        }
    }

    public async Task<JobStatus?> GetJobStatusAsync(string jobId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/api/status/{jobId}");
            if (!response.IsSuccessStatusCode)
            {
                return null;
            }
            
            return await response.Content.ReadFromJsonAsync<JobStatus>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting job status from Python service");
            return null;
        }
    }

    public async Task<byte[]?> GetPdfReportAsync(string jobId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/api/results/{jobId}/pdf");
            if (!response.IsSuccessStatusCode)
            {
                return null;
            }
            
            return await response.Content.ReadAsByteArrayAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting PDF report from Python service");
            return null;
        }
    }

    public async Task<AnalysisResults?> GetAnalysisDataAsync(string jobId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/api/results/{jobId}/data");
            if (!response.IsSuccessStatusCode)
            {
                return null;
            }
            
            return await response.Content.ReadFromJsonAsync<AnalysisResults>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting analysis data from Python service");
            return null;
        }
    }

    public async Task<JobStatus> UploadFileAsync(IFormFile file)
    {
        try
        {
            using var content = new MultipartFormDataContent();
            using var fileStream = file.OpenReadStream();
            using var streamContent = new StreamContent(fileStream);
            streamContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(file.ContentType);
            content.Add(streamContent, "file", file.FileName);

            var response = await _httpClient.PostAsync("/api/upload", content);
            response.EnsureSuccessStatusCode();
            
            var result = await response.Content.ReadFromJsonAsync<JobStatus>();
            if (result == null)
            {
                throw new Exception("Failed to deserialize response from Python service");
            }
            
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error uploading file to Python service");
            throw;
        }
    }

    public async Task<string> SendChatMessageAsync(string jobId, string question, bool includeHistory = true)
    {
        try
        {
            var request = new { question, include_history = includeHistory };
            var response = await _httpClient.PostAsJsonAsync($"/api/results/{jobId}/chat", request);
            response.EnsureSuccessStatusCode();
            
            var result = await response.Content.ReadFromJsonAsync<Dictionary<string, JsonElement>>();
            if (result != null && result.TryGetValue("answer", out var answerElement))
            {
                return answerElement.GetString() ?? "No response from chatbot";
            }
            return "No response from chatbot";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending chat message to Python service");
            throw;
        }
    }

    public async Task<List<string>> GetChatSuggestionsAsync(string jobId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/api/results/{jobId}/chat/suggestions");
            response.EnsureSuccessStatusCode();
            
            var result = await response.Content.ReadFromJsonAsync<Dictionary<string, JsonElement>>();
            if (result != null && result.TryGetValue("suggestions", out var suggestionsElement))
            {
                var suggestions = JsonSerializer.Deserialize<List<string>>(suggestionsElement.GetRawText());
                return suggestions ?? new List<string>();
            }
            return new List<string>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting chat suggestions from Python service");
            return new List<string>();
        }
    }

    public async Task<bool> ClearChatHistoryAsync(string jobId)
    {
        try
        {
            var response = await _httpClient.DeleteAsync($"/api/results/{jobId}/chat/history");
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error clearing chat history");
            return false;
        }
    }

    public async Task<bool> CheckHealthAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/health");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
}

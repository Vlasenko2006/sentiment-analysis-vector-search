using System.Text;
using System.Diagnostics;

namespace SentimentAPI.Middleware;

public class RequestResponseLoggingMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<RequestResponseLoggingMiddleware> _logger;
    private const string ComponentId = "DOTNET-API-GATEWAY";
    private const string ProgramName = "SentimentController.cs";

    public RequestResponseLoggingMiddleware(RequestDelegate next, ILogger<RequestResponseLoggingMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        var stopwatch = Stopwatch.StartNew();
        var requestId = Guid.NewGuid().ToString("N")[..8];
        
        // Log incoming request
        await LogRequest(context.Request, requestId);

        // Capture response
        var originalBodyStream = context.Response.Body;
        using var responseBody = new MemoryStream();
        context.Response.Body = responseBody;

        try
        {
            await _next(context);
            stopwatch.Stop();

            // Log outgoing response
            await LogResponse(context.Response, requestId, stopwatch.ElapsedMilliseconds);
        }
        finally
        {
            await responseBody.CopyToAsync(originalBodyStream);
        }
    }

    private async Task LogRequest(HttpRequest request, string requestId)
    {
        request.EnableBuffering();
        var body = await ReadStreamAsync(request.Body);
        request.Body.Position = 0;

        var logDir = Path.Combine(Directory.GetCurrentDirectory(), "debuglogs");
        Directory.CreateDirectory(logDir);
        var logFile = Path.Combine(logDir, $"dotnet-api-{DateTime.Now:yyyy-MM-dd}.log");

        var logEntry = new StringBuilder();
        logEntry.AppendLine($"[{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ss.fffZ}] [{ComponentId}] ← INPUT: HTTP Request");
        logEntry.AppendLine($"RequestId: {requestId}");
        logEntry.AppendLine($"Program: {ProgramName}");
        logEntry.AppendLine($"Method: {request.Method}");
        logEntry.AppendLine($"Path: {request.Path}{request.QueryString}");
        logEntry.AppendLine($"Headers: {string.Join(", ", request.Headers.Select(h => $"{h.Key}={h.Value}"))}");
        logEntry.AppendLine($"Body: {body}");
        logEntry.AppendLine($"MessagePath: frontend→.NET API");
        logEntry.AppendLine();

        await File.AppendAllTextAsync(logFile, logEntry.ToString());

        _logger.LogInformation("[{ComponentId}] ← INPUT: {Method} {Path} | RequestId: {RequestId}", 
            ComponentId, request.Method, request.Path, requestId);
    }

    private async Task LogResponse(HttpResponse response, string requestId, long elapsedMs)
    {
        response.Body.Position = 0;
        var body = await ReadStreamAsync(response.Body);
        response.Body.Position = 0;

        var logDir = Path.Combine(Directory.GetCurrentDirectory(), "debuglogs");
        var logFile = Path.Combine(logDir, $"dotnet-api-{DateTime.Now:yyyy-MM-dd}.log");

        var logEntry = new StringBuilder();
        logEntry.AppendLine($"[{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ss.fffZ}] [{ComponentId}] → OUTPUT: HTTP Response");
        logEntry.AppendLine($"RequestId: {requestId}");
        logEntry.AppendLine($"Program: {ProgramName}");
        logEntry.AppendLine($"StatusCode: {response.StatusCode}");
        logEntry.AppendLine($"Duration: {elapsedMs}ms");
        logEntry.AppendLine($"Body: {body}");
        logEntry.AppendLine($"MessagePath: .NET API→{(response.StatusCode >= 200 && response.StatusCode < 300 ? "Python Service" : "frontend (error)")}");
        logEntry.AppendLine();

        await File.AppendAllTextAsync(logFile, logEntry.ToString());

        _logger.LogInformation("[{ComponentId}] → OUTPUT: {StatusCode} | Duration: {Duration}ms | RequestId: {RequestId}", 
            ComponentId, response.StatusCode, elapsedMs, requestId);
    }

    private static async Task<string> ReadStreamAsync(Stream stream)
    {
        stream.Position = 0;
        using var reader = new StreamReader(stream, Encoding.UTF8, leaveOpen: true);
        var content = await reader.ReadToEndAsync();
        stream.Position = 0;
        return content;
    }
}

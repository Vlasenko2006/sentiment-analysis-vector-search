using Microsoft.Extensions.Diagnostics.HealthChecks;
using SentimentAPI.Services;

namespace SentimentAPI.Services;

public class PythonServiceHealthCheck : IHealthCheck
{
    private readonly IPythonServiceClient _pythonService;
    private readonly ILogger<PythonServiceHealthCheck> _logger;

    public PythonServiceHealthCheck(IPythonServiceClient pythonService, ILogger<PythonServiceHealthCheck> logger)
    {
        _pythonService = pythonService;
        _logger = logger;
    }

    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var isHealthy = await _pythonService.CheckHealthAsync();
            
            if (isHealthy)
            {
                return HealthCheckResult.Healthy("Python service is responsive");
            }
            
            return HealthCheckResult.Unhealthy("Python service is not responding");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Health check failed");
            return HealthCheckResult.Unhealthy("Python service health check failed", ex);
        }
    }
}

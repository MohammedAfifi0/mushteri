# Railway Scaling Guide for Concurrent Calls

## Problem: Calls Not Being Picked Up

When handling multiple concurrent calls, Railway needs proper configuration to scale horizontally.

## Solution: Configure Railway for Horizontal Scaling

### Step 1: Set Minimum Replicas

1. Go to your Railway project dashboard
2. Click on your service
3. Go to **Settings** → **Scaling**
4. Set **Min Replicas** to **1** (or higher for more capacity)
   - This ensures at least 1 container is always running
   - Prevents cold starts

### Step 2: Enable Auto-Scaling (Optional)

1. In the same **Scaling** section
2. Enable **Auto-scaling** if available
3. Set **Max Replicas** based on your needs:
   - **5 replicas** = ~500 concurrent calls (100 per container)
   - **10 replicas** = ~1000 concurrent calls

### Step 3: Configure Resource Limits

1. Go to **Settings** → **Resource Limits**
2. Set appropriate CPU and Memory:
   - **CPU**: 2-4 vCPU (for better concurrency)
   - **Memory**: 2-4 GB (for multiple concurrent calls)

### Step 4: Set Environment Variables (Optional)

Add these to your Railway environment variables if you want to tune concurrency:

```
LIMIT_CONCURRENCY=100
```

This sets max concurrent connections per container (default: 100).

### Step 5: Health Check Configuration

1. Go to **Settings** → **Healthcheck**
2. Set **Healthcheck Path** to: `/`
3. This keeps containers alive and prevents them from stopping

## How It Works

- **Single Container**: Handles up to 100 concurrent WebSocket connections
- **Multiple Containers**: Railway creates additional containers when traffic increases
- **Load Balancing**: Railway automatically distributes calls across containers
- **Auto-scaling**: Containers scale up/down based on traffic

## Testing Concurrent Calls

1. **Test with 2-3 calls simultaneously**
2. **Monitor Railway logs** to see if multiple containers are created
3. **Check metrics** in Railway dashboard for CPU/memory usage

## Troubleshooting

### Issue: Calls still not being picked up

**Solution:**
1. Check Railway logs for errors
2. Verify min replicas is set to 1+
3. Check resource limits (may need more CPU/memory)
4. Verify health check is configured

### Issue: High latency with many calls

**Solution:**
1. Increase CPU allocation (2-4 vCPU)
2. Increase memory allocation (2-4 GB)
3. Increase min replicas to 2-3 for baseline capacity

### Issue: Containers restarting

**Solution:**
1. Check logs for memory errors
2. Increase memory allocation
3. Check for unhandled exceptions in code

## Cost Optimization

- **Development**: Min replicas = 0 (scale to zero when idle)
- **Production**: Min replicas = 1-2 (always ready)
- **High Traffic**: Min replicas = 3-5 (handle spikes)

## Monitoring

1. **Railway Dashboard**: Check metrics for CPU, memory, network
2. **Logs**: Monitor for errors or connection issues
3. **Twilio Console**: Check call logs for connection failures

## Expected Performance

With proper configuration:
- ✅ **1 container**: ~100 concurrent calls
- ✅ **5 containers**: ~500 concurrent calls
- ✅ **10 containers**: ~1000 concurrent calls

Each container handles WebSocket connections independently, allowing true horizontal scaling.

---

## Quick Checklist

- [ ] Min replicas set to 1+
- [ ] Health check path set to `/`
- [ ] CPU: 2-4 vCPU
- [ ] Memory: 2-4 GB
- [ ] Test with 2-3 concurrent calls
- [ ] Monitor Railway logs and metrics


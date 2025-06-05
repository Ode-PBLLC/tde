# Simple Deployment for Sample Datasets

## Dataset Strategy (No Large Files)

### Datasets to Deploy:
1. **Solar facilities demo** - 924KB ✅
2. **GIST policy data** - 11MB ✅ 
3. **LSE policy modules** - 1MB ✅
4. **TZ-SAM analysis** - 11MB ✅

**Total: ~25MB** - Perfect for t3.medium!

## Simple MCP Server Architecture

### Add New Dataset Servers:

```python
# New MCP servers for additional datasets
# mcp/gist_server.py
# mcp/lse_policy_server.py  
# mcp/tz_sam_analysis_server.py
```

### Updated mcp_chat.py connections:
```python
async with MultiServerClient() as client:
    # Existing servers
    await client.connect_to_server("kg", "mcp/cpr_kg_server.py")
    await client.connect_to_server("solar", "mcp/solar_facilities_server.py") 
    await client.connect_to_server("formatter", "mcp/response_formatter_server.py")
    
    # New dataset servers
    await client.connect_to_server("gist", "mcp/gist_server.py")
    await client.connect_to_server("lse", "mcp/lse_policy_server.py")
    await client.connect_to_server("analysis", "mcp/tz_sam_analysis_server.py")
```

## File Organization

```
/opt/climate-api/app/
├── data/
│   ├── gist/gist.xlsx (11MB)
│   ├── lse/*.xlsx (1MB total)
│   ├── solar_facilities_demo.csv (924KB)
│   └── tz_sam_analysis.csv (11MB)
├── static/maps/ (Generated GeoJSON cache)
└── mcp/ (All MCP servers)
```

## t3.medium Configuration

### Optimized for Small Datasets:
```bash
# EC2 Instance: t3.medium
# - 2 vCPU, 4GB RAM
# - $30/month
# - Perfect for your actual data size

# Storage: 20GB EBS (plenty of room)
# Network: Moderate (sufficient for API + static files)
```

### Memory Usage Estimate:
- **Base system:** 1GB
- **FastAPI + dependencies:** 800MB  
- **6 MCP servers:** 1.2GB
- **All datasets loaded:** 100MB
- **GeoJSON generation:** 200MB temporary
- **Available headroom:** 700MB

### Simple Nginx Config:
```nginx
# No special optimizations needed
# Standard reverse proxy + static files
# Default timeouts are sufficient
```

## Deployment Steps (Simplified)

1. **EC2 t3.medium** - Launch instance
2. **Basic setup** - Python, dependencies, Nginx
3. **Deploy datasets** - Copy your ~25MB of data
4. **Start services** - Standard systemd setup
5. **Test endpoints** - All datasets accessible

## Cost Breakdown:
- **EC2 t3.medium:** $30/month
- **EBS 20GB:** $2/month  
- **Data transfer:** ~$1/month
- **Total:** ~$33/month

## Performance Expectations:
- **Cold start:** <5 seconds (all datasets loaded)
- **Map generation:** 1-3 seconds (reasonable dataset sizes)
- **Concurrent users:** 3-5 simultaneous (sufficient for development/demo)
- **Disk usage:** <5GB total

## Next Steps:
1. ✅ Use existing deployment scripts
2. ✅ Deploy on t3.medium  
3. ✅ Add new MCP servers for other datasets as needed
4. ✅ Monitor actual usage and scale if needed

**Bottom line:** With sample datasets only, t3.medium is perfect and cost-effective! 🎯
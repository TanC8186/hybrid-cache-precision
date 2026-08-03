import json, glob, sys
d = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/bench_lat/int4'
fs = sorted(glob.glob(d + '/*.json'))
if not fs:
    print('NO_JSON in', d); sys.exit(0)
for f in fs:
    j = json.load(open(f))
    print('FILE', f.split('/')[-1])
    print('  rate=%.3f  req_s=%.4f  out_tok_s=%.1f  tot_tok_s=%.1f  duration=%.1fs  completed=%d failed=%d  peak_conc=%d' % (
        j.get('request_rate'), j.get('request_throughput',0), j.get('output_throughput',0),
        j.get('total_token_throughput',0), j.get('duration',0), j.get('completed',0), j.get('failed',0),
        j.get('max_concurrent_requests',0)))
    print('  TTFT mean=%.1f p99=%.1f  TPOT mean=%.2f p99=%.2f  ITL mean=%.2f p99=%.2f' % (
        j.get('mean_ttft_ms',0), j.get('p99_ttft_ms',0), j.get('mean_tpot_ms',0),
        j.get('p99_tpot_ms',0), j.get('mean_itl_ms',0), j.get('p99_itl_ms',0)))

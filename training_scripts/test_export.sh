MAX_PIXELS=112896*2
ADAPTERS=$1
swift export \
    --adapters ${ADAPTERS} \
    --merge_lora true
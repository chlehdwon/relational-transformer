cd ../rustler
for db in rel-amazon rel-avito rel-event rel-f1 rel-hm rel-stack rel-trial; do
    # pixi run cargo run --release -- pre $db
    CUDA_VISIBLE_DEVICES=0,1,2,3 pixi run python -m rt.embed $db
done
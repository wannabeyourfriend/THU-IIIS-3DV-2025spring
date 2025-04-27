
if [ $# -ne 3 ]; then
    echo "用法: $0 <物体列表> <分辨率列表> <lambda_reg列表>"
    echo "示例: $0 \"apple bunny cup\" \"64 70 80 90 100 128\" \"0.1 0.5 1 2\""
    exit 1
fi

OBJECTS=($1)
RESOLUTIONS=($2)
LAMBDAS=($3)

CONFIG_FILE="e/GITHUB/3DV/PA/PA2/Problem3/configs/config.yaml"

for obj in "${OBJECTS[@]}"; do
    for res in "${RESOLUTIONS[@]}"; do
        for lambda in "${LAMBDAS[@]}"; do
            echo "正在运行配置: 物体=$obj, 分辨率=$res, lambda_reg=$lambda"
            
            sed -i.bak \
                -e "s/point_cloud:.*/point_cloud: \"$obj\"/" \
                -e "s/grid_res:.*/grid_res: $res/" \
                -e "s/lambda_reg:.*/lambda_reg: $lambda/" \
                $CONFIG_FILE

            python src/train.py --config-path configs/ --config-name config
            
            mv $CONFIG_FILE.bak $CONFIG_FILE
        done
    done
done

echo "所有参数组合训练完成！"
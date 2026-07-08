$ErrorActionPreference = "Stop"

$runs = @(
    @{
        Output = "outputs\paper_fp32_ieee_cifar_clean"
        Checkpoint = "outputs\task2\checkpoints\cifar\cifar_resnet20_clean_best.pt"
    },
    @{
        Output = "outputs\paper_fp32_ieee_cifar_nat_ft_grid"
        Checkpoint = "outputs\task2\checkpoints\cifar\cifar_resnet20_nat_ft_best.pt"
    },
    @{
        Output = "outputs\paper_fp32_ieee_cifar_nat_scratch_grid"
        Checkpoint = "outputs\task2\checkpoints\cifar\cifar_resnet20_nat_scratch_best.pt"
    },
    @{
        Output = "outputs\paper_fp32_ieee_cifar_robustsel_grid"
        Checkpoint = "outputs\task2\checkpoints\cifar\cifar_resnet20_nat_ft_robustsel_best.pt"
    },
    @{
        Output = "outputs\paper_fp32_ieee_cifar_per_occ_best"
        Checkpoint = "outputs\paper_cifar_per_occ_nat_scratch_full_e80\checkpoints\cifar\cifar_per_occ_nat_scratch_full_e80_best.pt"
    },
    @{
        Output = "outputs\paper_fp32_ieee_cifar_per_occ_last"
        Checkpoint = "outputs\paper_cifar_per_occ_nat_scratch_full_e80\checkpoints\cifar\cifar_per_occ_nat_scratch_full_e80_last.pt"
    }
)

foreach ($run in $runs) {
    Write-Host "[$(Get-Date -Format o)] FP32 IEEE CIFAR $($run.Output)"
    & python scripts\evaluate_cifar_fp32_ieee_bitserial.py `
        --output-dir $run.Output `
        --checkpoint $run.Checkpoint `
        --test-subset 0 `
        --batch-size 256 `
        --workers 0 `
        --random-repeats 10 `
        --fp32-repeats 3 `
        --mantissa-bits 23
}

Write-Host "[$(Get-Date -Format o)] DONE"

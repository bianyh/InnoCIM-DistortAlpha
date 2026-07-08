$ErrorActionPreference = "Stop"

$cifarRuns = @(
    @{
        Output = "outputs\paper_fpc_cifar_clean_full"
        Checkpoint = "outputs\task2\checkpoints\cifar\cifar_resnet20_clean_best.pt"
    },
    @{
        Output = "outputs\paper_fpc_cifar_nat_ft_full"
        Checkpoint = "outputs\task2\checkpoints\cifar\cifar_resnet20_nat_ft_best.pt"
    },
    @{
        Output = "outputs\paper_fpc_cifar_nat_scratch_full"
        Checkpoint = "outputs\task2\checkpoints\cifar\cifar_resnet20_nat_scratch_best.pt"
    },
    @{
        Output = "outputs\paper_fpc_cifar_nat_robustsel_full"
        Checkpoint = "outputs\task2\checkpoints\cifar\cifar_resnet20_nat_ft_robustsel_best.pt"
    },
    @{
        Output = "outputs\paper_fpc_cifar_per_occ_scratch_full_best"
        Checkpoint = "outputs\paper_cifar_per_occ_nat_scratch_full_e80\checkpoints\cifar\cifar_per_occ_nat_scratch_full_e80_best.pt"
    },
    @{
        Output = "outputs\paper_fpc_cifar_per_occ_scratch_full_last"
        Checkpoint = "outputs\paper_cifar_per_occ_nat_scratch_full_e80\checkpoints\cifar\cifar_per_occ_nat_scratch_full_e80_last.pt"
    }
)

foreach ($run in $cifarRuns) {
    Write-Host "[$(Get-Date -Format o)] CIFAR $($run.Output)"
    & python scripts\evaluate_cifar_bitserial_fixedpoint.py `
        --output-dir $run.Output `
        --checkpoint $run.Checkpoint `
        --test-subset 0 `
        --batch-size 256 `
        --workers 0 `
        --random-repeats 10 `
        --bits "1,2,3,4,5,6,7,8"
}

$voxRuns = @(
    @{
        Output = "outputs\paper_fpc_voxforge_clean"
        Checkpoint = "outputs\task2\checkpoints\voxforge\voxforge_crnn_clean_best.pt"
    },
    @{
        Output = "outputs\paper_fpc_voxforge_nat_ft"
        Checkpoint = "outputs\task2\checkpoints\voxforge\voxforge_crnn_nat_ft_best.pt"
    },
    @{
        Output = "outputs\paper_fpc_voxforge_nat_scratch"
        Checkpoint = "outputs\task2\checkpoints\voxforge\voxforge_crnn_nat_scratch_best.pt"
    }
)

foreach ($run in $voxRuns) {
    Write-Host "[$(Get-Date -Format o)] VOX $($run.Output)"
    & python scripts\evaluate_voxforge_bitserial_fixedpoint.py `
        --output-dir $run.Output `
        --checkpoint $run.Checkpoint `
        --random-repeats 10 `
        --bits "1,2,3,4,5,6,7,8" `
        --batch-size 16 `
        --workers 0
}

Write-Host "[$(Get-Date -Format o)] DONE"

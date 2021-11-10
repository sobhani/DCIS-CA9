import pathlib

from sccnn_classifier import featuremap_visualization

if __name__ == '__main__':
    opts = {
        'exp_dir': str(pathlib.Path(
            r'D:\Shan\MyCodes\TracerX\CellClassification\Code\20171019-SCCNNClassifier\ExpDir-IHC')),
        'data_dir': str(pathlib.Path(
            r'R:\tracerx\tracerx100\IHC_diagnostic\Misc\Annotations-180308\cell_labels-180308\celllabels\LTX012.ndpi/')),
        'cws_dir': str(pathlib.Path(
            r'T:\tracerx100\IHC_diagnostic\data\cws\LTX012.ndpi/'))
    }

    featuremap_visualization.run(opts_in=opts)

src=en
tgt=de
#TEXT=/mnt/zhangyuhao/DATA/mergewmt-ende-silent
#TEXT=/mnt/zhangyuhao/DATA/mustc-enfr/data/mustc-silent
TEXT=/mnt/zhangyuhao/DATA/mergewmt-enes-silent
root=/mnt/zhangyuhao/fairseq-0.12.3
tag=mergewmt-enes-silent
output=data-bin/$tag
#python3 $root/fairseq_cli/preprocess.py --source-lang $src --target-lang $tgt --trainpref $TEXT/merge.spm.filter --validpref $TEXT/dev.spm --testpref $TEXT/test.spm --destdir $output --workers 12 --srcdict $TEXT/share.dict --joined-dictionary
#python3 $root/fairseq_cli/preprocess.py --source-lang $src --target-lang $tgt --trainpref $TEXT/train.spm.silent --validpref $TEXT/dev.spm --testpref $TEXT/test.spm  --destdir $output --workers 12 --joined-dictionary --srcdict /mnt/zhangyuhao/DATA/mustc-enfr/data/mustc-silent/share.dict
python3 $root/fairseq_cli/preprocess.py --source-lang $src --target-lang $tgt --trainpref $TEXT/merge.spm.filter --validpref $TEXT/dev.spm --testpref $TEXT/test.spm  --destdir $output --workers 12 --joined-dictionary --srcdict $TEXT/dict.es.txt

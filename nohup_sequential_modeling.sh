nohup /root/no-error/bin/python3 /workspace/shopping-classification/multilayer_classifier.py train ./data/train ./model/train b &> /workspace/shopping-classification/log/nohup_b.out&
echo "b finish!"
nohup /root/no-error/bin/python3 /workspace/shopping-classification/multilayer_classifier.py train ./data/train ./model/train m &> /workspace/shopping-classification/log/nohup_m.out&
echo "m finish!"
nohup /root/no-error/bin/python3 /workspace/shopping-classification/multilayer_classifier.py train ./data/train ./model/train s &> /workspace/shopping-classification/log/nohup_s.out&
echo "s finish!"
nohup /root/no-error/bin/python3 /workspace/shopping-classification/multilayer_classifier.py train ./data/train ./model/train d &> /workspace/shopping-classification/log/nohup_d.out&
echo "d finish!"
echo "all process finish!"
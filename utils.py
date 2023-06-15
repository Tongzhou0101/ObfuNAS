from thop import profile
from network import *
from setting import *
import warnings
warnings.filterwarnings("ignore")


def acc_avg(spec):
  _, computed_metrics = nasbench.get_metrics_from_spec(spec)
  val_total = 0
  test_total = 0
  for i in range(3):
    val_total += computed_metrics[108][i]['final_validation_accuracy']
    test_total += computed_metrics[108][i]['final_test_accuracy']
  return val_total/3, test_total/3


def fitness(spec):
  val_acc, test_acc = acc_avg(spec)
  input = torch.randn(1, 3, 32, 32)
  net = Network(spec)
  flops, params = profile(net, inputs=(input, ))
  # print('ratio',ratio)
  # print('acc',val_acc)
  score = -val_acc
  # print('score',score)
  return score, flops, val_acc, test_acc


def info(spec):
  _, computed_metrics = nasbench.get_metrics_from_spec(spec)
  input = torch.randn(1, 3, 32, 32)
  net = Network(spec)
  flops, params = profile(net, inputs=(input, ))

  return flops, computed_metrics[108][0]['final_test_accuracy']


def check(spec, popu):
  if len(popu) == 0:
    flag = 1
  else:
    for i in range(len(popu)):
      if len(spec.original_matrix) != len(popu[i].original_ops):
          flag = 1
      elif np.any(spec.original_matrix!=popu[i].original_matrix) or spec.original_ops!=popu[i].original_ops:
          flag = 1
      else:
          flag = 0
          break
  return flag



def check_same(spec, popu):
  for i in range(len(popu)):
    if len(spec.original_matrix) != len(popu[i][1].original_ops):
        flag = 1
    elif np.any(spec.original_matrix!=popu[i][1].original_matrix) or spec.original_ops!=popu[i][1].original_ops:
        flag = 1
    else:
        flag = 0
        break
  return flag

if __name__=='__main__':
    # Query an Inception-like cell from the dataset.
    victim_spec = api.ModelSpec(
        matrix = [[0, 1, 0, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0],],  # output layer
        # Operations at the vertices of the module, matches order of matrix.
        ops = [INPUT, CONV3X3, CONV3X3, CONV3X3, CONV3X3, OUTPUT])


    #victim_val_avg, victim_test_avg = acc_avg(victim_spec)
    print(info(victim_spec))
    #print(victim_val_avg, victim_test_avg)

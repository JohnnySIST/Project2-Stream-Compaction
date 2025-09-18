const CPU = [0.0016, 0.054, 1.7093, 52.4132, 105.806, 206.849, 424.237, 816.657, 1662.63];
const NaiveGPU = [1.68691, 1.84899, 4.20902, 86.1861, 171.778, 353.931, 703.445, 3192.93, 14334];
const EfficientGPU = [1.38016, 1.37459, 2.68237, 27.3095, 52.3393, 102.907, 204.032, 407.458, 2523.06];
const Thrust = [0.981888, 1.06662, 1.34141, 3.1232, 5.23088, 8.9111, 16.5718, 78.364, 404.013];

const N = [10, 15, 20, 25, 26, 27, 28, 29, 30];
const N_values = N.map(v => v);
const offset= 3;

function logAndOffset(arr, offset) {
  return arr.map(v => Math.log10(v) + offset);
}

option = {
  title: {
    text: 'Scan Performance'
  },
  legend: { data: ['CPU', 'Naive GPU', 'Efficient GPU', 'Thrust'] },
  xAxis: { type: 'value', name: 'log10(N)', min: 10, max: 30 },
  yAxis: { type: 'value', name: 'log10(Time)' },
  series: [
    { name: 'CPU', type: 'line', data: N_values.map((x,i) => [x, logAndOffset(CPU, offset)[i]]) },
    { name: 'Naive GPU', type: 'line', data: N_values.map((x,i) => [x, logAndOffset(NaiveGPU, offset)[i]]) },
    { name: 'Efficient GPU', type: 'line', data: N_values.map((x,i) => [x, logAndOffset(EfficientGPU, offset)[i]]) },
    { name: 'Thrust', type: 'line', data: N_values.map((x,i) => [x, logAndOffset(Thrust, offset)[i]]) },
  ]
};
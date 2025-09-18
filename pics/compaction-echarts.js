const CPU_without_scan = [0.0025, 0.067, 2.2095, 70.2928, 143.941, 282.771, 558.38, 1003.12, 2273.97];
const CPU_with_scan = [0.0042, 0.1381, 4.8973, 156.767, 323.207, 652.236, 1376.53, 2541.53, 6898.14];
const Efficient_GPU = [0.206528, 0.471424, 1.34582, 33.5974, 65.5928, 132.566, 281.205, 2543.02, 11972.5];

const N = [10, 15, 20, 25, 26, 27, 28, 29, 30];
const N_values = N.map(v => v);
const offset= 3;

function logAndOffset(arr, offset) {
  return arr.map(v => Math.log10(v) + offset);
}

option = {
  title: {
    text: 'Compaction Performance'
  },
  legend: { data: ['CPU_without_scan', 'CPU_with_scan', 'Efficient_GPU'] },
  xAxis: { type: 'value', name: 'log10(N)', min: 10, max: 30 },
  yAxis: { type: 'value', name: 'log10(Time)' },
  series: [
    { name: 'CPU_without_scan', type: 'line', data: N_values.map((x,i) => [x, logAndOffset(CPU_without_scan, offset)[i]]) },
    { name: 'CPU_with_scan', type: 'line', data: N_values.map((x,i) => [x, logAndOffset(CPU_with_scan, offset)[i]]) },
    { name: 'Efficient_GPU', type: 'line', data: N_values.map((x,i) => [x, logAndOffset(Efficient_GPU, offset)[i]]) },
  ]
};
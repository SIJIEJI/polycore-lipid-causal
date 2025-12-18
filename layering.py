const csvContent = await window.fs.readFile('merged_data.csv', { encoding: 'utf8' });

import Papa from 'papaparse';
const data = Papa.parse(csvContent, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true
});

console.log("数据概览:");
console.log("总行数:", data.data.length);
console.log("列名:", data.meta.fields);

// 检查前几行数据
console.log("\n前3行数据:");
data.data.slice(0, 3).forEach((row, idx) => {
    console.log(`第${idx+1}行:`, row);
});

// 检查PatientID分布
import _ from 'lodash';
const patientCounts = _.countBy(data.data, 'PatientID');
console.log("\n每个患者的样本数:");
console.log(patientCounts);
console.log("患者总数:", Object.keys(patientCounts).length);

// 检查关键变量的分布
const keyVars = ['Sweat CH (uM)', 'Sweat TG (uM)', 'Sweat Rate (uL/min)', 'Total cholesterol (mg/dL)', 'TG (mg/dL)'];
console.log("\n关键变量统计:");
keyVars.forEach(varName => {
    const values = data.data.map(row => row[varName]).filter(v => v != null);
    console.log(`${varName}: 均值=${(values.reduce((a,b)=>a+b,0)/values.length).toFixed(2)}, 范围=[${Math.min(...values).toFixed(2)}, ${Math.max(...values).toFixed(2)}]`);
});
Output

Result

数据概览:
总行数: 115
列名: ["PatientID","Glucose (mg/dL)","HDL C (mg/dL)","TG (mg/dL)","Total cholesterol (mg/dL)","LDL Chol (mg/dL)","Sweat Rate (uL/min)","Sweat CH (uM)","Sweat TG (uM)","Sweat CH (uM)_mean","Sweat CH (uM)_std","Sweat CH (uM)_min","Sweat CH (uM)_max","Sweat Rate (uL/min)_mean","Sweat Rate (uL/min)_std","Sweat Rate (uL/min)_min","Sweat Rate (uL/min)_max","Total cholesterol (mg/dL)_mean","Total cholesterol (mg/dL)_std","Total cholesterol (mg/dL)_min","Total cholesterol (mg/dL)_max","Age (18>)","Gender","HgA1C","Glucose","Blood Pressure H","Blood Pressure L","Pulse","Weight (lb)","CALCULATED BMI","BMR (kcal)","Fat%","Fat mass (lb)","FFM (lb)","Predicted muscle mass (lb)","TBW (lb)"]

前3行数据:
第1行: {"PatientID":1,"Glucose (mg/dL)":97.44805194805194,"HDL C (mg/dL)":42.64285714285714,"TG (mg/dL)":218.99516426541697,"Total cholesterol (mg/dL)":200.6666666666667,"LDL Chol (mg/dL)":114.22477667072619,"Sweat Rate (uL/min)":3.22361111111111,"Sweat CH (uM)":3.145082323013093,"Sweat TG (uM)":127.13470681458003,"Sweat CH (uM)_mean":1.8297311757402004,"Sweat CH (uM)_std":1.1067376155410618,"Sweat CH (uM)_min":0.641574383460696,"Sweat CH (uM)_max":3.145082323013093,"Sweat Rate (uL/min)_mean":2.9375555555555555,"Sweat Rate (uL/min)_std":0.2965664754868898,"Sweat Rate (uL/min)_min":2.5152777777777775,"Sweat Rate (uL/min)_max":3.22361111111111,"Total cholesterol (mg/dL)_mean":171.82222222222225,"Total cholesterol (mg/dL)_std":17.59030491948603,"Total cholesterol (mg/dL)_min":157.33333333333334,"Total cholesterol (mg/dL)_max":200.6666666666667,"Age (18>)":50,"Gender":0,"HgA1C":5.3,"Glucose":103,"Blood Pressure H":111,"Blood Pressure L":69,"Pulse":68,"Weight (lb)":184,"CALCULATED BMI":34.8,"BMR (kcal)":1413,"Fat%":45.9,"Fat mass (lb)":84.4,"FFM (lb)":99.6,"Predicted muscle mass (lb)":94.6,"TBW (lb)":73}
第2行: {"PatientID":1,"Glucose (mg/dL)":139.6818181818182,"HDL C (mg/dL)":31.90476190476191,"TG (mg/dL)":193.2290025674261,"Total cholesterol (mg/dL)":161.94444444444443,"LDL Chol (mg/dL)":91.3938820261973,"Sweat Rate (uL/min)":2.84,"Sweat CH (uM)":2.1002715498531206,"Sweat TG (uM)":104.9080824088748,"Sweat CH (uM)_mean":1.8297311757402004,"Sweat CH (uM)_std":1.1067376155410618,"Sweat CH (uM)_min":0.641574383460696,"Sweat CH (uM)_max":3.145082323013093,"Sweat Rate (uL/min)_mean":2.9375555555555555,"Sweat Rate (uL/min)_std":0.2965664754868898,"Sweat Rate (uL/min)_min":2.5152777777777775,"Sweat Rate (uL/min)_max":3.22361111111111,"Total cholesterol (mg/dL)_mean":171.82222222222225,"Total cholesterol (mg/dL)_std":17.59030491948603,"Total cholesterol (mg/dL)_min":157.33333333333334,"Total cholesterol (mg/dL)_max":200.6666666666667,"Age (18>)":50,"Gender":0,"HgA1C":5.3,"Glucose":103,"Blood Pressure H":111,"Blood Pressure L":69,"Pulse":68,"Weight (lb)":184,"CALCULATED BMI":34.8,"BMR (kcal)":1413,"Fat%":45.9,"Fat mass (lb)":84.4,"FFM (lb)":99.6,"Predicted muscle mass (lb)":94.6,"TBW (lb)":73}
第3行: {"PatientID":1,"Glucose (mg/dL)":125.43506493506493,"HDL C (mg/dL)":35.33333333333333,"TG (mg/dL)":238.4436613163044,"Total cholesterol (mg/dL)":176.22222222222226,"LDL Chol (mg/dL)":93.20015662562804,"Sweat Rate (uL/min)":3.2200000000000006,"Sweat CH (uM)":2.526776430684124,"Sweat TG (uM)":116.15213946117274,"Sweat CH (uM)_mean":1.8297311757402004,"Sweat CH (uM)_std":1.1067376155410618,"Sweat CH (uM)_min":0.641574383460696,"Sweat CH (uM)_max":3.145082323013093,"Sweat Rate (uL/min)_mean":2.9375555555555555,"Sweat Rate (uL/min)_std":0.2965664754868898,"Sweat Rate (uL/min)_min":2.5152777777777775,"Sweat Rate (uL/min)_max":3.22361111111111,"Total cholesterol (mg/dL)_mean":171.82222222222225,"Total cholesterol (mg/dL)_std":17.59030491948603,"Total cholesterol (mg/dL)_min":157.33333333333334,"Total cholesterol (mg/dL)_max":200.6666666666667,"Age (18>)":50,"Gender":0,"HgA1C":5.3,"Glucose":103,"Blood Pressure H":111,"Blood Pressure L":69,"Pulse":68,"Weight (lb)":184,"CALCULATED BMI":34.8,"BMR (kcal)":1413,"Fat%":45.9,"Fat mass (lb)":84.4,"FFM (lb)":99.6,"Predicted muscle mass (lb)":94.6,"TBW (lb)":73}

每个患者的样本数:
{"1":5,"2":5,"3":5,"4":5,"5":5,"6":5,"7":5,"9":5,"10":5,"11":5,"12":5,"13":5,"14":5,"15":5,"16":5,"17":5,"18":5,"19":5,"20":5,"21":5,"22":5,"23":5,"24":5}
患者总数: 23

关键变量统计:
Sweat CH (uM): 均值=2.22, 范围=[0.11, 5.50]
Sweat TG (uM): 均值=167.06, 范围=[1.44, 589.05]
Sweat Rate (uL/min): 均值=2.06, 范围=[0.20, 4.75]
Total cholesterol (mg/dL): 均值=146.21, 范围=[84.04, 208.99]
TG (mg/dL): 均值=139.15, 范围=[43.04, 337.64]

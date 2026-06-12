import React from 'react';
import { Slider, Row, Col, Typography } from 'antd';

const { Text } = Typography;

export interface ParamValues {
  temperature: number;
  max_per_industry: number;
  corr_threshold: number;
  min_score_ratio: number;
  bull_gate: number;
  top_n: number;
  select_k: number;
  turnover_penalty: number;
  persistence_bonus: number;
}

export const DEFAULT_PARAMS: ParamValues = {
  temperature: 0.5, max_per_industry: 2, corr_threshold: 0.8,
  min_score_ratio: 0.7, bull_gate: 0.6, top_n: 10,
  select_k: 5, turnover_penalty: 0.02, persistence_bonus: 0.01,
};

interface Props {
  values: ParamValues;
  onChange: (vals: ParamValues) => void;
}

const SLIDERS = [
  { key: 'temperature' as const, label: '温度 (Temperature)', min: 0.1, max: 2.0, step: 0.1, tip: '越低权重越集中' },
  { key: 'max_per_industry' as const, label: '行业上限', min: 1, max: 5, step: 1, tip: '每个行业最多选几只' },
  { key: 'corr_threshold' as const, label: '相关性阈值', min: 0.5, max: 1.0, step: 0.05, tip: '超过则惩罚得分' },
  { key: 'min_score_ratio' as const, label: '最低得分比', min: 0.3, max: 0.9, step: 0.05, tip: '低于Top-1的百分比则不选' },
  { key: 'bull_gate' as const, label: '牛市门槛 (Bull Gate)', min: 0.0, max: 1.0, step: 0.05, tip: 'Bull Strength低于此值空仓' },
  { key: 'top_n' as const, label: '候选池大小 (Top-N)', min: 5, max: 30, step: 1, tip: '初选股票数量' },
  { key: 'select_k' as const, label: '持仓数量 (Select-K)', min: 1, max: 10, step: 1, tip: '最终组合股票数' },
  { key: 'turnover_penalty' as const, label: '换手惩罚', min: 0.0, max: 0.1, step: 0.005, tip: '新开仓得分惩罚' },
  { key: 'persistence_bonus' as const, label: '持仓奖励', min: 0.0, max: 0.05, step: 0.005, tip: '已有持仓得分奖励' },
];

const ParamSliders: React.FC<Props> = ({ values, onChange }) => {
  const update = (key: keyof ParamValues, val: number) => onChange({ ...values, [key]: val });

  return (
    <div>
      {SLIDERS.map((s) => (
        <Row key={s.key} style={{ marginBottom: 16 }}>
          <Col span={24}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text strong>{s.label}</Text>
              <Text type="secondary">{values[s.key]}</Text>
            </div>
            <Slider min={s.min} max={s.max} step={s.step} value={values[s.key]} onChange={(v) => update(s.key, v as number)} />
            <Text type="secondary" style={{ fontSize: 11 }}>{s.tip}</Text>
          </Col>
        </Row>
      ))}
    </div>
  );
};

export default ParamSliders;

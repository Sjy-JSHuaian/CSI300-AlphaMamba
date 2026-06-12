import React, { useState } from 'react';
import { ConfigProvider, Tabs, Layout, Typography } from 'antd';
import { BarChartOutlined, LineChartOutlined, ExperimentOutlined, SettingOutlined, HistoryOutlined } from '@ant-design/icons';
import PredictTab from './components/PredictTab';
import CompareTab from './components/CompareTab';
import TuningTab from './components/TuningTab';
import BacktestTab from './components/BacktestTab';
import HistoryTab from './components/HistoryTab';

const { Header, Content } = Layout;
const { Title } = Typography;

const tabItems = [
  { key: 'predict', label: '预测', icon: <BarChartOutlined />, children: <PredictTab /> },
  { key: 'compare', label: '对比', icon: <ExperimentOutlined />, children: <CompareTab /> },
  { key: 'tuning', label: '调参', icon: <SettingOutlined />, children: <TuningTab /> },
  { key: 'backtest', label: '回测', icon: <LineChartOutlined />, children: <BacktestTab /> },
  { key: 'history', label: '历史', icon: <HistoryOutlined />, children: <HistoryTab /> },
];

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('predict');

  return (
    <ConfigProvider
      theme={{
        token: { colorPrimary: '#1677ff', borderRadius: 8 },
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ background: '#001529', display: 'flex', alignItems: 'center', padding: '0 24px' }}>
          <Title level={3} style={{ color: '#fff', margin: 0 }}>
            CSI300 AlphaMamba · 分析工作台
          </Title>
        </Header>
        <Content style={{ padding: 24 }}>
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={tabItems}
            size="large"
            style={{ background: '#fff', padding: '0 24px', borderRadius: 8 }}
          />
        </Content>
      </Layout>
    </ConfigProvider>
  );
};

export default App;

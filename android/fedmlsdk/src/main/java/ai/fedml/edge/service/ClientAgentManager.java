package ai.fedml.edge.service;

import android.text.TextUtils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;

import org.json.JSONObject;

import ai.fedml.edge.OnTrainProgressListener;
import ai.fedml.edge.OnTrainingStatusListener;
import ai.fedml.edge.service.communicator.EdgeCommunicator;
import ai.fedml.edge.service.communicator.OnMLOpsMsgListener;
import ai.fedml.edge.service.communicator.OnTrainStartListener;
import ai.fedml.edge.service.communicator.OnTrainStopListener;
import ai.fedml.edge.service.communicator.message.MessageDefine;
import ai.fedml.edge.service.component.TokenChecker;
import ai.fedml.edge.service.component.MetricsReporter;
import ai.fedml.edge.utils.LogHelper;
import ai.fedml.edge.utils.preference.SharePreferencesData;

import androidx.annotation.NonNull;

public final class ClientAgentManager implements MessageDefine {
    private final OnTrainProgressListener onTrainProgressListener;
    private final OnTrainingStatusListener onTrainingStatusListener;
    private final EdgeCommunicator edgeCommunicator;
    private final TokenChecker mTokenChecker;
    private final MetricsReporter mReporter;
    private volatile long mEdgeId = 0;
    private final Gson mGson;

    private ClientManager mClientManager;

    public ClientAgentManager(@NonNull final OnTrainingStatusListener onTrainingStatusListener,
                              @NonNull final OnTrainProgressListener onTrainProgressListener) {
        this.onTrainingStatusListener = onTrainingStatusListener;
        this.onTrainProgressListener = onTrainProgressListener;
        edgeCommunicator = EdgeCommunicator.getInstance();
        mTokenChecker = new TokenChecker(SharePreferencesData.getBindingId());
        mReporter = MetricsReporter.getInstance();
        mReporter.setTrainingStatusListener(onTrainingStatusListener);
        mGson = new GsonBuilder().setPrettyPrinting().create();
        SharePreferencesData.clearHyperParameters();
        bindCommunicator(SharePreferencesData.getBindingId());
    }

    public void bindCommunicator(final String bindId) {
        if (!TextUtils.isEmpty(bindId)) {
            try {
                mEdgeId = Long.parseLong(bindId);
            } catch (NumberFormatException e) {
                LogHelper.e(e, "bindCommunicator bindId(%s) parseInt Exception", bindId);
                return;
            }
            // register train message receive handlers
            registerMessageReceiveHandlers(mEdgeId);
        } else {
            LogHelper.wtf("bindCommunicator failed. Maybe bindId is empty.");
        }
    }

    /**
     * register handlers
     *
     * @param edgeId edge id
     */
    public void registerMessageReceiveHandlers(final long edgeId) {
        mReporter.reportTrainingStatus(edgeId, KEY_CLIENT_STATUS_IDLE);
        // 接受到训练开始的信息
        final String startTrainTopic = "flserver_agent/" + edgeId + "/start_train";
        edgeCommunicator.subscribe(startTrainTopic, (OnTrainStartListener) this::handleTrainStart);
        // 停止训练消息监听
        final String stopTrainTopic = "flserver_agent/" + edgeId + "/stop_train";
        edgeCommunicator.subscribe(stopTrainTopic, (OnTrainStopListener) this::handleTrainStop);
        // listen to message from MLOps
        final String MLOpsQueryStatusTopic = "/mlops/report_device_status";
        edgeCommunicator.subscribe(MLOpsQueryStatusTopic, (OnMLOpsMsgListener) this::handleMLOpsMsg);
    }

    private void handleTrainStart(JSONObject msgParams) {
        LogHelper.d("onStartTrain: %s", msgParams.toString());
        if (mEdgeId == 0) {
            // 未绑定或解绑状态不能启动训练
            return;
        }
        //TODO: authentic
        final String groupId = msgParams.optString(GROUP_ID, "");
        boolean isAuth = mTokenChecker.authentic(groupId);
        if (!isAuth) {
            LogHelper.d("handleTrainStart authentic failed.");
            return;
        }
        // TODO: waiting dataset split, then download the dataset package and Training Client App

        long runId = msgParams.optLong("runId", 0);

        JSONObject hyperParameters = null;
        JSONObject runConfigJson = msgParams.optJSONObject(RUN_CONFIG);
        if (runConfigJson != null) {
            hyperParameters = runConfigJson.optJSONObject(HYPER_PARAMETERS_CONFIG);
            if (hyperParameters != null) {
                JsonElement jsonElement = JsonParser.parseString(hyperParameters.toString());
                SharePreferencesData.saveHyperParameters(mGson.toJson(jsonElement));
            } else {
                SharePreferencesData.clearHyperParameters();
            }
            onTrainingStatusListener.onStatusChanged(KEY_CLIENT_STATUS_INITIALIZING);
        }
        // Launch Training Client
        mClientManager = new ClientManager(mEdgeId, runId, hyperParameters, onTrainProgressListener);
    }

    private void handleTrainStop(JSONObject msgParams) {
        LogHelper.d("handleTrainStop :%s", msgParams.toString());
        mReporter.reportTrainingStatus(mEdgeId, KEY_CLIENT_STATUS_STOPPING);
//        edgeCommunicator.unsubscribe("flserver_agent/" + mEdgeId + "/start_train");
//        edgeCommunicator.unsubscribe("flserver_agent/" + mEdgeId + "/stop_train");
        mReporter.reportTrainingStatus(mEdgeId, KEY_CLIENT_STATUS_IDLE);

        // Stop Training Client
        mClientManager.stopTrain();
        mClientManager = null;
    }

    private void handleMLOpsMsg(JSONObject msgParams) {
        LogHelper.d("handleMLOpsMsg :%s", msgParams.toString());
        mReporter.reportClientActiveStatus(mEdgeId);
    }
}

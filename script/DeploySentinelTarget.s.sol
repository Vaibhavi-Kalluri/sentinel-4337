// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Script} from "forge-std/Script.sol";
import {SentinelTarget} from "../src/SentinelTarget.sol";

contract DeploySentinelTarget is Script {
    function run() external returns (SentinelTarget target) {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");

        vm.startBroadcast(deployerPrivateKey);
        target = new SentinelTarget();
        vm.stopBroadcast();
    }
}

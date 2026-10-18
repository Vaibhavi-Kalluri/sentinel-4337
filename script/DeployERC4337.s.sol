// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Script} from "forge-std/Script.sol";
import {EntryPoint} from "account-abstraction/core/EntryPoint.sol";
import {SimpleAccountFactory} from "account-abstraction/accounts/SimpleAccountFactory.sol";

contract DeployERC4337 is Script {
    function run() external returns (EntryPoint entryPoint, SimpleAccountFactory factory) {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");

        vm.startBroadcast(deployerPrivateKey);

        entryPoint = new EntryPoint();
        factory = new SimpleAccountFactory(entryPoint);

        vm.stopBroadcast();
    }
}
